# Benchmarking

Benchmark 的目标不是只看 loss，而是区分三件事：

1. `small real-data smoke`: 用教程真实 `.bin/.idx` 数据验证训练栈。
2. `data pipeline cost`: 用不同规模真实 `.bin/.idx` 数据比较吞吐变化。
3. `quality benchmark`: 用 validation/evaluation 判断模型是否变好。

## Throughput benchmark

先跑教程真实数据 smoke test：

```bash
DATA_SPLIT=100,0,0 EVAL_ITERS=0 \
bash scripts/21_run_pretrain_real_4gpu.sh runs/data/the_verdict_text_document
```

再跑更大的真实抽样或正式数据：

```bash
bash scripts/21_run_pretrain_real_4gpu.sh runs/data/your_corpus_text_document
```

提取日志：

```bash
python scripts/30_parse_training_log.py runs/logs/pretrain_real.log
```

## What good looks like

小数据 smoke test 只能证明链路可跑，不能代表硬件上限。正式数据吞吐慢，问题通常在：

1. dataset cache 构建。
2. filesystem I/O。
3. `num-workers` 太小或太大。
4. 数据文件太碎。
5. tokenizer/preprocess 没有提前做，训练时仍在阻塞。

## Hardware utilization

运行中观察：

```bash
nvidia-smi dmon -s pucm
```

重点看：

1. GPU utilization 是否长期接近满载。
2. memory 是否接近 OOM。
3. power 是否接近卡的上限。
4. 多卡 utilization 是否均衡。

## Quality benchmark

`LLMs-from-scratch` 和 `reasoning-from-scratch` 的质量评估包括：

1. held-out `validation loss`。
2. instruction response spot check。
3. classification accuracy / F1。
4. `MATH-500` exact/verifier accuracy。
5. `MMLU` multiple-choice accuracy。
6. `LLM-as-a-judge`。
7. leaderboard aggregation such as `Elo` or `Bradley-Terry`。

本教程在 `docs/09_evaluation.md` 给出统一流程。
