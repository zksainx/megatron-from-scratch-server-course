# Data Pipeline

`LLMs-from-scratch` 从小文本开始解释 `tokenization`、`sliding window` 和 `DataLoader`。在 Megatron 中，等价流程被拆成两个阶段：

1. `raw data`: 你维护 JSONL / CSV / parquet / plain text。
2. `indexed dataset`: Megatron 读取 `.bin/.idx`，训练时按 `sequence length` 取样。

## Pretraining data format

Megatron `tools/preprocess_data.py` 的标准输入是 JSONL，每行至少有 `text` 字段：

```json
{"text": "Large language models predict the next token."}
{"text": "Training data is packed into fixed-length sequences."}
```

每行是一篇 `document`。加 `--append-eod` 后，文档末尾会追加 `EOD token`，帮助模型学习文档边界。

`pretraining` 数据教模型的不是“回答问题”，而是语言分布本身：什么 token 在什么上下文后更可能出现。loss 作用在几乎所有 non-padding token 上，所以每个 token 都在更新 embedding、attention pattern、MLP feature 和 output head。直观地说，模型在这里学习：

1. `syntax`: 局部 grammar、标点、引用、段落结构。
2. `semantics`: 词与概念的共现结构。
3. `discourse`: 长距离指代、叙事顺序、topic transition。
4. `compression prior`: 哪些 pattern 值得用参数记住，哪些 pattern 应该靠上下文推断。

`the-verdict.txt` 是极小文学语料，技术价值不在训练出好模型，而在让你观察完整 `causal LM objective` 如何从 raw text 变成 `.bin/.idx`，再被 Megatron 切成 fixed-length samples。这里最重要的 takeaway 是：`pretraining` 数据决定 base model 的 probability manifold；后面的 `SFT`、`RLVR`、`distillation` 都只是在这个 manifold 上重新加权或局部变形。

`EOD token` 很关键。没有清晰 document boundary 时，packing 可能把两篇无关文档拼接成一个上下文，模型会错误学习跨文档 continuation。对于大规模训练，`document` 粒度、去重、source mixture 和 `EOD` 策略会直接影响模型是否学到稳定的 long-context behavior。

## Instruction / SFT data format

Megatron 当前 `SFTDataset` 读取 JSONL，每行有 `messages` 字段：

```json
{"messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Explain gradient clipping."},{"role":"assistant","content":"Gradient clipping limits gradient norm to stabilize training."}]}
```

理想的 instruction SFT 通常会把 prompt token 的 label mask 掉，只对 assistant response 计算 loss；在 Megatron 中这取决于 `SFTTokenizer` 的 `prompt format` 和 tokenizer 是否带 chat template。

`SFT` 数据教模型的是 conditional policy：在给定 `system` 和 `user` 条件下，assistant 应该选择哪类 response。prompt token 仍参与 forward attention，因为它们提供条件；在带 chat template 的主流 tokenizer 上，prompt token 通常不参与 loss，因为我们不想训练模型“复述用户问题”，而是训练它在这些条件下生成 assistant side token。

这个 mask 细节非常重要。假设一条样本有 300 个 prompt tokens 和 50 个 response tokens，如果错误地对全部 350 个 token 计算 loss，梯度会主要被 prompt modeling 主导，模型学到的是“用户通常怎么提问”，而不是“assistant 应该怎么回答”。response-only loss 把优化目标集中到 action tokens 上，这和后续 `RLVR` 中只评价 rollout response 是一致的。

本教程的 `scripts/22_run_sft_4gpu.sh` 为了离线可跑，默认使用本地 GPT-2 tokenizer 和 `--sft-tokenizer-prompt-format identity`；这个教学配置不做 response-only mask，所有非 padding token 都参与 loss。它适合验证 SFT 数据和 Megatron SFT pipeline。正式 instruction tuning 应换成带 chat template 的 Llama/Qwen/Nemotron 类 tokenizer，并使用对应 prompt format。

`instruction-data.json` 的价值是让 base model 学会 role-conditioned generation：遵守任务边界、按要求输出、在 instruction 和 optional input 之间建立条件依赖。它不会凭空注入大量新知识；它主要改变 decoding prior，让模型从“继续文本”转向“完成任务”。

## Classification data as SFT

`LLMs-from-scratch` Ch 6 使用 `classification head` 做 spam classification。Megatron GPT 主线不专门加 classification head。服务器实践中更通用的做法是把分类任务转成 generative `SFT`：

```json
{"messages":[{"role":"system","content":"Classify the message as spam or not_spam."},{"role":"user","content":"You won a free prize. Click now!"},{"role":"assistant","content":"spam"}]}
```

评估时用 exact match / accuracy / F1 统计 assistant 输出。

把 classification 转成 generative `SFT` 本质上是把一个 discriminative decision boundary 投影到 language modeling space。模型不是输出一个 classifier head 的 logits，而是在固定 label vocabulary 上生成 token。这样做的技术后果是：

1. label tokenization 会影响难度，例如 `not_spam` 是一个 token 还是多个 token。
2. prompt wording 会改变 label prior，所以 evaluation prompt 必须固定。
3. generation decoding 需要受控，通常使用 greedy 或 very low temperature。
4. metric 要在 normalized string 上计算，避免 `"spam."` 和 `"spam"` 被误判。

takeaway：classification-as-generation 牺牲了一些专用 classifier 的校准性，但换来统一的 GPT training stack。它适合教学和统一 serving；如果你需要高质量 calibrated probability，应额外做 logprob scoring 或专门 classifier calibration。

## Reasoning data

`reasoning-from-scratch` 的 reasoning trace 可以直接放到 assistant response：

```json
{"messages":[{"role":"system","content":"Solve math problems. Put final answer in \\boxed{}."},{"role":"user","content":"What is 17+25?"},{"role":"assistant","content":"17+25=42. Therefore, \\boxed{42}."}]}
```

如果要训练短回答模型，可以把 `chain-of-thought` 留给 teacher generation，然后 distill 成更短 response。

reasoning 数据教模型的不是普通事实，而是 latent computation pattern。把 reasoning trace 放进 assistant response，相当于让模型在 token space 中学习一条可观察的 intermediate trajectory：先分解问题，再执行推理，再规范化 final answer。这个过程改变的是模型的 computation allocation：遇到数学或逻辑 prompt 时，更倾向于生成 scratchpad token，而不是直接跳到答案。

这里有一个关键取舍：长 `chain-of-thought` 提供 dense supervision，但也会让模型学会冗长、模板化甚至错误的 reasoning style。短 answer distillation 则压缩输出长度，但会减少中间步骤监督。更实际的做法是保留两类数据：

1. `reasoning SFT`: 教会模型展开推理过程。
2. `distilled short answer`: 教会模型在部署时更短、更稳定地输出。

`reasoning-from-scratch` 的样本在本教程里承担这两个角色：`math_train_sample.json` 让你构造 supervised reasoning examples，`sample_ollama_outputs.json` 让你观察 teacher-generated traces 如何进入 distillation pipeline。

## Dataset quality checklist

1. `deduplication`: instruction tuning 前先去重，避免 benchmark contamination。
2. `length histogram`: 检查 token length，决定 `seq-length`。
3. `split`: train/valid/test 固定随机种子，避免调参污染 test。
4. `format validation`: JSONL 必须一行一个 JSON object。
5. `license`: 真实训练前确认数据授权。

更深一层看，数据质量不是“文本干不干净”这么简单，而是 gradient quality。每个 batch 都在消耗 GPU 时间把某种分布偏置写进参数。你要追问：

1. 这批 token 的 loss 是否落在真正想优化的行为上。
2. 这批数据是否和 evaluation distribution 对齐。
3. 这批数据是否和已有能力冲突，导致 catastrophic forgetting。
4. 长样本是否挤占 token budget，却只提供很少有效监督。
5. mixture ratio 是否让小但关键的数据源被大语料淹没。

对于 `SFT` 和 reasoning，样本数不是唯一尺度，response token 数和有效 loss token 数更重要。1000 条短 answer 样本可能只有几万 loss tokens；1000 条 reasoning trace 可能有几十万 loss tokens，并且会强烈塑造输出风格。因此比较数据集时要看 `loss tokens per capability`，而不是只看 row count。

## From-scratch tutorial datasets

主线数据来自两个教程仓库：

| Dataset | Source | 用途 |
|---|---|---|
| `the-verdict.txt` | `LLMs-from-scratch/ch02/01_main-chapter-code` | `pretraining` text corpus |
| `instruction-data.json` | `LLMs-from-scratch/ch07/01_main-chapter-code` | instruction `SFT` |
| `instruction-data-with-response.json` | `LLMs-from-scratch/ch07/01_main-chapter-code` | instruction evaluation |
| `instruction-data-with-preference.json` | `LLMs-from-scratch/ch07/04_preference-tuning-with-dpo` | `DPO` concept / preference data |
| `math500_test.json` | `reasoning-from-scratch/ch03/01_main-chapter-code` | reasoning evaluation |
| `math_train_sample.json` | `reasoning-from-scratch/ch08/02_generate_distillation_data` | reasoning SFT/distillation sample |
| `sample_ollama_outputs.json` | `reasoning-from-scratch/ch08/02_generate_distillation_data` | teacher-output distillation sample |

## What each generated file teaches

`scripts/10_prepare_from_scratch_datasets.py` 会把教程数据转成几种 Megatron-friendly artifacts。它们的作用不同，不应该混在一个“训练数据”概念里理解。

| Generated file | Training/eval role | It teaches or measures |
|---|---|---|
| `pretrain_the_verdict.jsonl` | `pretraining` corpus | 教 base model 做 next-token prediction，学习 narrative text distribution、document boundary 和 LM pipeline |
| `sft_instruction_data.jsonl` | instruction `SFT` | 教 conditional assistant behavior：读 instruction、结合 input、输出 target response |
| `eval_instruction_responses.jsonl` | instruction evaluation | 不训练；用来比较 generated response、reference response 和已有 model response |
| `preference_dpo_format.jsonl` | preference / `DPO` concept | 表达 pairwise preference：同一 prompt 下 chosen 比 rejected 更好 |
| `eval_math500.jsonl` | reasoning evaluation | 不训练；测 final answer correctness、parser/verifier 和 benchmark contamination |
| `sft_reasoning_math_train_sample.jsonl` | reasoning `SFT` sample | 教模型在数学 prompt 下输出 answer format，样本极少，只适合 pipeline demonstration |
| `sft_reasoning_distillation_sample.jsonl` | distillation `SFT` sample | 教模型模仿 teacher reasoning trace 或 teacher final answer style |

这里最容易犯的错误是把 evaluation set 当作 training data。`eval_math500.jsonl` 和 `eval_instruction_responses.jsonl` 的主要价值是 measurement，不是 supervision。把它们混入 SFT 会让指标虚高，并破坏你对模型真实泛化能力的判断。

`preference_dpo_format.jsonl` 也不等价于普通 SFT。SFT 只告诉模型“这个 answer 是 target”；preference data 告诉模型“在两个 answer 之间，哪个更符合偏好”。这类信号更接近 ranking constraint，通常需要 `DPO`、`IPO`、`ORPO` 或 RL-style objective，而不是简单把 `chosen` 拼成 assistant response 就结束。简单 SFT on chosen 可以作为 warmup，但会丢掉 rejected sample 中携带的 negative signal。

## Capability-oriented view

从能力角度看，数据管线应该这样理解：

```text
raw text
  -> base distribution / world-text prior
instruction pairs
  -> controllable assistant policy
classification labels
  -> constrained decision behavior
reasoning traces
  -> explicit intermediate computation style
preference pairs
  -> relative quality ordering
eval sets
  -> measurement, not optimization
```

这个视角比“文件格式”更重要。文件格式只决定 loader 能不能读；数据的语义角色决定 optimizer 会把模型推向哪里。

生成 Megatron-friendly JSONL：

```bash
python scripts/10_prepare_from_scratch_datasets.py \
  --train-root /sgl-workspace/zkx/train \
  --output-dir data/from_scratch
```

输出：

```text
data/from_scratch/pretrain_the_verdict.jsonl
data/from_scratch/sft_instruction_data.jsonl
data/from_scratch/eval_instruction_responses.jsonl
data/from_scratch/preference_dpo_format.jsonl
data/from_scratch/eval_math500.jsonl
data/from_scratch/sft_reasoning_math_train_sample.jsonl
data/from_scratch/sft_reasoning_distillation_sample.jsonl
```
