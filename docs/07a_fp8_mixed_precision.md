# Ch 7a — FP8 Mixed Precision Training

本章覆盖 FP8 混合精度训练的完整知识链：数值格式、scaling 策略、Blackwell SM100 的硬件原生 MXFP8 支持，以及在 Megatron-LM 中的实操配置。

前置知识：已完成 Ch 5 (pretraining) 和 Ch 7 (benchmarking)。

---

## 1. 为什么用 FP8

BF16 是当前大模型训练的默认精度。FP8 把每个元素从 16 bit 压缩到 8 bit，理论上可以获得：

- **2× GEMM 吞吐**：Tensor Core 在相同时钟周期内处理两倍数据
- **~50% 权重内存节省**：如果参数也以 FP8 存储
- **更低的通信量**：分布式训练中 all-gather / reduce-scatter 数据量减半

代价是精度损失——8 bit 能表示的数值范围和精度远小于 16 bit，需要 **scaling 策略** 来弥补。

> **注意**：本教程的教学模型 (12 层 / 768 hidden) 非常小，训练过程以 memory-bound 为主而非 compute-bound。因此你可能 **看不到明显的吞吐提升**。本章的目标是理解机制、掌握配置方法，为大模型训练做准备。

---

## 2. FP8 数值格式

IEEE 尚未标准化 FP8，但业界有两种主流格式：

| 格式 | 结构 | 动态范围 | 精度 | 用途 |
|------|------|----------|------|------|
| **E4M3** | 1 sign + 4 exponent + 3 mantissa | ±448 | 较高 (8 级尾数) | 权重、前向激活 |
| **E5M2** | 1 sign + 5 exponent + 2 mantissa | ±57344 | 较低 (4 级尾数) | 反向梯度 |

对比 BF16 (1+8+7)：E4M3 的指数位少 4 bit，动态范围从 ~3.4×10³⁸ 缩小到 448。一个异常大的值就能让 E4M3 溢出，这就是 scaling 存在的原因。

Megatron 中通过 `--fp8-format` 选择：
- `e4m3`：前向和反向都用 E4M3
- `hybrid`（推荐）：前向用 E4M3，反向梯度用 E5M2（更宽的动态范围保护梯度）

---

## 3. Scaling 策略

FP8 的核心挑战：如何把 BF16 范围的数值映射到 FP8 的窄范围内而不溢出/下溢。答案是 **缩放因子 (scale factor)**——在量化前除以 scale，在反量化后乘回来。

不同的策略决定了 scale 的粒度（一个 scale 覆盖多少个元素）和计算时机。

### 3.1 Per-tensor Delayed Scaling (`delayed`)

```
--fp8-format hybrid
```
（`--fp8-recipe` 默认就是 `delayed`）

- **粒度**：整个 tensor 共享一个 FP32 scale
- **时机**：维护一个 amax history 窗口，用历史最大绝对值估算下一步的 scale
- **参数**：`--fp8-amax-history-len`（窗口长度）、`--fp8-amax-compute-algo`（`max` 或 `most_recent`）
- **缺点**：一个 outlier 影响整个 tensor 的 scale；需要跨 DP rank 同步 amax
- **硬件**：Hopper (SM90) 及以上

这是 FP8 训练最早的方案，TransformerEngine 1.x 时代的默认选择。

### 3.2 Per-tensor Current Scaling (`tensorwise`)

```
--fp8-format hybrid --fp8-recipe tensorwise
```

- **粒度**：同 delayed，整个 tensor 一个 scale
- **时机**：使用当前 tensor 的实际 amax（无延迟），不需要 history
- **优势**：比 delayed 更稳定（无 lag），实现更简单
- **要求**：TransformerEngine ≥ 2.2.0

### 3.3 Block Scaling (`blockwise`)

```
--fp8-format e4m3 --fp8-recipe blockwise
```

- **粒度**：类似 DeepSeek-V3 方案——激活按 1×128 block，权重按 128×128 block
- **Scale 类型**：FP32
- **优势**：细粒度 scale，outlier 只影响局部 block
- **要求**：TransformerEngine ≥ 2.3.0

DeepSeek-V3 用此类方案在 Hopper 上成功训练了 671B MoE 模型，loss 偏差 < 0.25%（相对 BF16）。

### 3.4 MXFP8 Block Scaling (`mxfp8`) — Blackwell 原生

```
--fp8-format hybrid --fp8-recipe mxfp8
```

- **粒度**：每 **32 个连续元素** 共享一个 E8M0 scale（power-of-2）
- **标准**：OCP Microscaling (MX) v1.0 规范
- **硬件支持**：SM100 (Blackwell) Tensor Core 原生指令 `tcgen05.mma...block_scale`

这就是你说的 **"1d1d"**：每个 GEMM 操作数沿一个维度做 1D block scaling，两个操作数各自独立缩放。

**为什么 MXFP8 是 Blackwell 上的优先候选**：

| 特性 | Delayed / Tensorwise | MXFP8 |
|------|---------------------|-------|
| Scale 粒度 | 整个 tensor | 32 元素 |
| Scale 格式 | FP32 | E8M0 (power-of-2) |
| 硬件加速 | 软件管理 | **Tensor Core 原生** |
| 跨 GPU 同步 | 需要 amax all-reduce | **不需要** |
| Outlier 鲁棒性 | 差（一个 outlier 影响整个 tensor） | **好（只影响 32 个元素）** |
| 端到端加速 | 取决于模型和实现 | 论文和厂商实验报告过 1.28×-1.37×，小模型不保证 |

> **术语对照**："1d1d" = "1-dimensional data, 1-dimensional scale"，即每个操作数做一维 block scaling。在 Megatron/TransformerEngine 中，这个概念对应 `MXFP8BlockScaling` recipe。OCP MX 标准定义 block size = 32，scale format = E8M0。

---

## 4. Scaling 策略对比

| Recipe | Block Size | Scale 格式 | 需要 amax 同步 | Min TE | Min SM | 推荐场景 |
|--------|-----------|-----------|---------------|--------|--------|---------|
| `delayed` | 整个 tensor | FP32 | 是 | 1.0 | SM89 | 遗留代码，Hopper |
| `tensorwise` | 整个 tensor | FP32 | 否 | 2.2.0 | SM89 | Hopper，简单稳定 |
| `blockwise` | 1×128 / 128×128 | FP32 | 否 | 2.3.0 | SM89 | DeepSeek-V3 风格，MoE |
| **`mxfp8`** | **1×32** | **E8M0** | **否** | **2.1.0** | **SM100** | **Blackwell，推荐** |

---

## 5. 工业实践

### DeepSeek-V3

DeepSeek-V3 (671B MoE) 是首个大规模使用细粒度 FP8 训练的公开模型：
- 激活：1×128 tile scaling
- 权重：128×128 block scaling
- 全部使用 E4M3（包括梯度）
- 在 Hopper 上用自定义 GEMM kernel 实现 FP32 累加（绕过 H100 的 14-bit 累加限制）
- 关键经验：embedding 和 output head 保持 BF16

### NVIDIA MXFP8 推荐配置

NVIDIA 在 "Recipes for Pre-training LLMs with MXFP8" 中验证了：
- 使用 `MXFP8BlockScaling` + E4M3 风格的 MXFP8 配方
- Scale 计算使用 ceiling rounding（向上取整），不用 OCP v1.0 建议的 floor
- 第一层和最后一层保持 BF16
- 在论文覆盖的模型、数据和实现上匹配 BF16 收敛；吞吐收益依赖模型规模和 kernel 覆盖率

### Cursor MXFP8 Kernel

Cursor 团队在 Blackwell 上编写自定义 MXFP8 kernel，公开分享过 MoE 训练加速经验。这里的关键点不是照搬数字，而是：Blackwell 的内存层次与 Hopper 不同，直接移植 Hopper kernel 不一定有效。

---

## 6. Megatron-LM FP8 参数参考

### 核心参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--fp8-format` | None | 启用 FP8。`e4m3` 或 `hybrid`（推荐） |
| `--fp8-recipe` | `delayed` | Scaling recipe: `delayed`/`tensorwise`/`mxfp8`/`blockwise`/`custom` |
| `--fp8-margin` | 0 | Scaling factor 计算的 margin |
| `--fp8-amax-history-len` | 1 | amax history 窗口长度（仅 `delayed`） |
| `--fp8-amax-compute-algo` | `most_recent` | amax 算法：`max` 或 `most_recent`（仅 `delayed`） |
| `--fp8-wgrad` | True | 权重梯度是否用 FP8 计算 |

### 精度控制

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--first-last-layers-bf16` | False | 首尾 N 层保持 BF16（**不兼容 `delayed`**） |
| `--num-layers-at-start-in-bf16` | 1 | BF16 起始层数 |
| `--num-layers-at-end-in-bf16` | 1 | BF16 结尾层数 |
| `--fp8-dot-product-attention` | False | 注意力计算用 FP8 |
| `--fp8-multi-head-attention` | False | 多头注意力用 FP8 |

### 参数存储与通信

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--fp8-param-gather` | False | 参数以 FP8 存储和 all-gather（需要 `--use-distributed-optimizer`） |
| `--reuse-grad-buf-for-mxfp8-param-ag` | False | MXFP8 + FP8 param gather 时复用 grad buffer，降低额外显存 |
| `--tp-only-amax-red` | False | 仅在 TP/TP-CP domain 内同步 amax |

### 约束关系

- `--fp8-param-gather` 需要 `--use-distributed-optimizer` 或 FSDP2 或 inference mode
- `--first-last-layers-bf16` 不能与 `--fp8-recipe delayed` 一起使用（delayed 使用全局 fp8_autocast，无法逐层切换精度）
- `--fp8-param-gather` + optimizer CPU offload 仅支持 `delayed` recipe
- MXFP8 大模型训练推荐同时评估 `--fp8-param-gather --reuse-grad-buf-for-mxfp8-param-ag`；本章教学 preset 默认不启用，是为了减少 checkpoint/参数存储行为变化，先验证 activation/weight GEMM 的 MXFP8 路径。

---

## 7. 实操：在教程模型上启用 FP8

### 7.1 确认环境

```bash
bash scripts/00_healthcheck.sh
# 确认输出包含:
# transformer_engine_torch_ok True
# megatron_have_te True
```

确认 GPU 支持 MXFP8：
```bash
python -c "import torch; print('SM version:', torch.cuda.get_device_capability())"
# 应输出 (10, 0) 或更高
```

### 7.2 准备数据

如果还没有预处理过的数据：
```bash
python scripts/10_prepare_from_scratch_datasets.py \
  --train-root /sgl-workspace/zkx/train --output-dir data/from_scratch

bash scripts/11_preprocess_gpt_data.sh \
  data/from_scratch/pretrain_the_verdict.jsonl runs/data/the_verdict
```

### 7.3 运行 BF16 基线

```bash
DATA_SPLIT=100,0,0 EVAL_ITERS=0 \
  bash scripts/21_run_pretrain_real_4gpu.sh runs/data/the_verdict_text_document
```

### 7.4 运行 FP8 MXFP8 训练

```bash
DATA_SPLIT=100,0,0 EVAL_ITERS=0 \
  bash scripts/25_run_pretrain_fp8_4gpu.sh runs/data/the_verdict_text_document
```

脚本与 BF16 版本的唯一区别：
1. 额外 source `configs/4gpu_edu_fp8_mxfp8.env`，引入 `PRECISION_ARGS` 和 `OPTIMIZER_ARGS`
2. `--bf16` 替换为 `$PRECISION_ARGS`（展开为 `--bf16 --fp8-format hybrid --fp8-recipe mxfp8 --first-last-layers-bf16`）
3. 添加 `$OPTIMIZER_ARGS`（展开为 `--use-distributed-optimizer`）
4. 输出目录使用 `pretrain_fp8/`（不覆盖 BF16 结果）

如果你在同一个 `RUN_ROOT` 下切换了 `SEQ_LENGTH`、`HIDDEN_SIZE`、层数或并行配置，先换一个新的 `RUN_ROOT` 或 checkpoint 子目录；Megatron 默认会从 `--load "$CKPT_DIR"` 恢复，形状不匹配的旧 checkpoint 会导致启动失败。

### 7.5 对比结果

```bash
python scripts/30_parse_training_log.py runs/logs/pretrain_real.log
python scripts/30_parse_training_log.py runs/logs/pretrain_fp8.log
```

关注指标：
- `throughput_tflops`：FP8 理论上更高（小模型可能差异不大）
- `lm_loss`：两者应该接近（FP8 略有波动是正常的）
- `grad_norm`：FP8 下梯度 norm 可能略有不同

---

## 8. 预期效果

| 场景 | 预期吞吐变化 | 原因 |
|------|-------------|------|
| 教学模型 (12L/768H, seq=512) | 可能无显著变化 | 模型太小，训练 bottleneck 在内存带宽和 kernel launch，不在 GEMM 计算 |
| 中等模型 (32L/4096H, seq=4096) | +20-37% | GEMM 占比增大，FP8 加速生效 |
| 大模型 (64L+/8192H+, seq=8192+) | +30-40%+ | compute-bound，FP8 GEMM 2× 吞吐充分发挥 |

本章的目标不是追求加速，而是：
1. 理解 FP8 数值格式和 scaling 机制
2. 掌握 Megatron-LM FP8 配置方法
3. 验证 FP8 训练 pipeline 在你的硬件上可以正确运行
4. 为后续大模型训练建立配置基础

---

## 9. 稳定性注意事项

FP8 训练的精度更低，以下措施可以提高稳定性：

1. **梯度裁剪**：`--clip-grad 1.0`（教程默认已启用）。FP8 下梯度更容易出现大值
2. **首尾层 BF16**：`--first-last-layers-bf16`。Embedding 和 output head 对精度敏感
3. **充分 warmup**：`--lr-warmup-fraction 0.01` 或更高。让 scaling factor 在训练早期稳定
4. **监控 grad norm**：`--log-params-norm` 已启用。如果 grad_norm 突然飙升，可能是 FP8 精度问题
5. **频繁 checkpoint**：及时保存，出现 NaN loss 时可以回滚
6. **排查顺序**：如果 FP8 训练出现问题，先用 BF16 重跑确认是否是数据/超参问题

### Fallback 方案

如果 `--fp8-recipe mxfp8` 在你的环境上不工作（例如 TE 版本或 driver 兼容性问题），可以退回到：

```bash
# Per-tensor current scaling (SM89+, TE >= 2.2.0)
PRECISION_ARGS="--bf16 --fp8-format hybrid --fp8-recipe tensorwise"

# Per-tensor delayed scaling (最广泛兼容)
PRECISION_ARGS="--bf16 --fp8-format hybrid --fp8-amax-history-len 1024 --fp8-amax-compute-algo max"
```

---

## 10. 配置文件说明

本章新增的配置文件 `configs/4gpu_edu_fp8_mxfp8.env` 叠加在基础配置 `configs/4gpu_edu_pretrain.env` 之上，只定义 FP8 相关变量：

```bash
# FP8 precision flags
PRECISION_ARGS="--bf16 --fp8-format hybrid --fp8-recipe mxfp8 --first-last-layers-bf16"

# Distributed optimizer (ZeRO Stage-1)
OPTIMIZER_ARGS="--use-distributed-optimizer"
```

另外，`configs/4gpu_b200_llama_fp8.env` 是面向大模型 (32L/4096H Llama) 的性能配置，额外启用了 `--fp8-param-gather`（将参数以 FP8 存储和通信）。
