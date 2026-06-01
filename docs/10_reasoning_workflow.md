# Reasoning Workflow

这章覆盖 `reasoning-from-scratch` Ch 1-7：从 base model 到 reasoning model 的完整路线。

## Base -> Instruct -> Reasoning

常见 pipeline：

```text
base pretraining
  -> instruction SFT
  -> reasoning SFT / distillation
  -> RLVR with GRPO
  -> inference-time scaling
  -> evaluation and selection
```

Megatron 在这里承担 `SFT`、`GRPO/RL` 和大规模 checkpoint 管理。`reasoning-from-scratch` 的 notebook 和脚本承担 evaluation/parser/prompting 思路。

## Inference-time scaling

不改模型参数，通过更多 test-time compute 提升结果：

1. `Chain-of-thought prompting`: 要求模型写 reasoning steps。
2. `self-consistency`: 同一题采样多次，用 majority vote 或 verifier 选答案。
3. `best-of-N`: 生成 N 个候选，用 scorer/verifier 选最好。
4. `self-refinement`: 让模型批判并修正自己的答案。

这些方法必须先跑，因为它们是 post-training 前的 baseline。

## RLVR / GRPO

`GRPO` 的核心对象：

| Term | Meaning |
|---|---|
| `policy` | 当前训练模型 |
| `rollout` | policy 生成的回答 |
| `reward` | verifier 或 rule-based scorer 给的分数 |
| `group` | 同一 prompt 的多个 samples |
| `advantage` | 相对 group 平均表现的优势 |
| `KL` | 限制 policy 偏离 reference |
| `clip ratio` | 防止 policy update 过大 |

本地 Megatron 有 `train_rl.py` 与 `examples/rl`。教学脚本 `scripts/23_run_grpo_skeleton_4gpu.sh` 提供 4 卡命令骨架，但真实 RL 需要可用 checkpoint、tokenizer、environment config 和数据集。

## Format reward

reasoning 模型常需要格式约束，例如最终答案写成：

```text
\boxed{42}
```

`format reward` 不评价数学正确性，只评价输出结构。它通常和 answer verifier reward 组合。

## KL and stability

如果 `KL beta` 太小，模型可能 reward hack；太大，学习太慢。教学上先跑：

```text
GRPO_KL_BETA=0.0   # 仅用于消融实验，不建议实际训练使用
GRPO_KL_BETA=0.01
GRPO_KL_BETA=0.05
```

比较 reward、length、format accuracy、MATH accuracy。

## Tracking

至少记录：

1. average reward。
2. response length。
3. format accuracy。
4. KL。
5. clip fraction。
6. pass@1 / pass@N（pass@k: k 个独立生成中至少 1 个正确的概率）。
7. sample outputs。

只看 reward 容易误判，因为 reward model 或 rule-based verifier 可能被 exploit。
