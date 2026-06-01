# Course Map

这一章定义本教程如何覆盖两个 `from scratch` 教程。左侧是原教程知识点，右侧是本教程中的服务器化训练任务。术语保留 English，说明使用中文。

## LLMs-from-scratch Coverage

| 原教程章节 | 核心知识 | 本教程覆盖位置 | Megatron 对应实践 |
|---|---|---|---|
| Ch 1: Understanding LLMs | `LLM lifecycle`、`pretraining`、`finetuning`、`generation` | `docs/00`, `docs/05` | 把 lifecycle 改成 server pipeline |
| Ch 2: Working with Text Data | `tokenization`、`BPE`、`sliding window`、`DataLoader`、`train/val/test split` | `docs/02`, `docs/03` | `tools/preprocess_data.py`、`.bin/.idx`、`--split` |
| Ch 3: Attention Mechanisms | `self-attention`、`causal mask`、`multi-head attention` | `docs/04` | `--num-attention-heads`、`--attention-backend flash/fused/auto` |
| Ch 4: GPT Model | `embedding`、`Transformer block`、`LayerNorm`、`position embedding`、`KV cache`、`GQA/MQA/MoE/SWA/MLA` | `docs/04`, `docs/10` | `--position-embedding-type rope`、`--group-query-attention`、`--num-query-groups`、inference scripts |
| Ch 5: Pretraining | `training loop`、`loss`、`optimizer`、`scheduler`、`checkpoint`、`generation` | `docs/05`, `docs/06` | `pretrain_gpt.py`、`--lr-decay-style cosine`、`--save/--load` |
| Ch 5 bonus: Gutenberg/data scale | raw corpus to pretraining data | `docs/02`, `docs/03` | JSONL corpus preparation and preprocessing |
| Ch 5 bonus: speed | `bf16`、`FlashAttention`、`torch.compile`、`DDP`、`vocab padding` | `docs/04`, `docs/07` | `bf16/fp8`、`tensor parallel`、`pipeline parallel`、`sequence parallel` |
| Ch 5 bonus: architectures | `Llama`、`Qwen`、`Gemma`、`Olmo`、`MoE` | `docs/04` | architecture flags rather than handwritten model classes |
| Ch 6: Classification finetuning | `classification dataset`、`classification head`、`accuracy/F1` | `docs/08`, `docs/09` | 转为 generative `SFT` 或外部 classifier benchmark |
| Ch 7: Instruction finetuning | `instruction dataset`、`prompt template`、`response loss mask` | `docs/08` | Megatron `SFTDataset` + `SFTTokenizer` |
| Ch 7 bonus: dataset generation | synthetic instruction data and filtering | `docs/08` | JSONL `messages` generator and dedup checklist |
| Ch 7 bonus: DPO | `preference dataset`、`chosen/rejected`、`preference optimization` | `docs/08` | 作为 post-training concept；Megatron 本地以 `GRPO` 为主要 RL path |
| Appendix A | PyTorch / GPU / DDP basics | `docs/01`, `docs/04` | `torchrun`、`NCCL`、rank/world size |
| Appendix D | better training loop | `docs/06` | Megatron logging, evaluation interval, checkpoint interval |
| Appendix E | `LoRA` / parameter-efficient finetuning | `docs/08` | 概念覆盖；Megatron 主线使用 full-rank SFT，LoRA 作为扩展路径 |

## reasoning-from-scratch Coverage

| 原教程章节 | 核心知识 | 本教程覆盖位置 | Megatron 对应实践 |
|---|---|---|---|
| Ch 1: Reasoning Models | `reasoning model`、base vs instruct vs reasoning | `docs/10` | post-training pipeline |
| Ch 2: Generate with pretrained LLM | `sampling`、`temperature`、`top-k/top-p`、chat loop | `docs/09`, `docs/10` | inference/evaluation scripts and prompt templates |
| Ch 3: Evaluate reasoning | `MATH-500`、answer parser、verifier | `docs/09` | bridge scripts call local verifier examples |
| Ch 4: Inference-time scaling | `CoT prompting`、`self-consistency` | `docs/10` | benchmark generated solutions before training changes |
| Ch 5: Self-refinement | `best-of-N`、`scorer`、iterative refinement | `docs/10` | post-training evaluation protocol |
| Ch 6: RLVR with GRPO | `reward`、`rollout`、`policy update`、`group advantage` | `docs/10` | `train_rl.py` and `examples/rl` configs |
| Ch 7: Improve GRPO | `clip ratio`、`KL`、`format reward`、tracking | `docs/10` | Megatron RL flags and logs |
| Ch 8: Distillation | `teacher generation`、`student SFT`、reasoning trace compression | `docs/11` | generate `messages` JSONL then run `SFT` |
| Appendix C: Qwen3 source | Qwen3 architecture | `docs/04`, `docs/10` | architecture flags and tokenizer/model conversion |
| Appendix D/E | larger LLMs and batching | `docs/04`, `docs/07` | `micro_batch_size`、`global_batch_size`、parallelism |
| Appendix F | `MMLU`、leaderboard、`LLM-as-a-judge` | `docs/09` | evaluation recipes and wrappers |
| Appendix G | chat interface | `docs/09`, `docs/12` | server inference endpoint or CLI |

## 本教程的取舍

1. 不再手写 `Transformer component`。所有组件通过 Megatron 参数、模型构建器和 checkpoint 表达。
2. 不把 Notebook 当作主入口。所有任务都可以用 `bash` 或 `python` 在服务器终端执行。
3. 知识覆盖保留完整，但实现重点变成 `data -> preprocess -> train -> log -> checkpoint -> benchmark -> evaluate -> post-train`。
4. `DPO`、`LoRA`、部分 architecture variants 在本教程中作为 concept 和 extension 覆盖；当前本地 Megatron 主线脚本优先支持 `pretraining`、`SFT`、`GRPO/RL`。
