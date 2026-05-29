# Evaluation

这章覆盖 `LLMs-from-scratch` 的 instruction evaluation、classification evaluation，以及 `reasoning-from-scratch` 的 `MATH-500`、`MMLU`、leaderboard、`LLM-as-a-judge`。

## Evaluation layers

| Layer | Question | Tooling |
|---|---|---|
| `validation loss` | 模型是否继续拟合语言分布 | Megatron eval loop |
| `generation sanity check` | 输出是否可读、是否遵循 prompt | inference script |
| `classification accuracy` | label 是否正确 | exact match / F1 |
| `instruction quality` | 回答是否有帮助 | local judge or API judge |
| `reasoning accuracy` | final answer 是否正确 | MATH parser/verifier |
| `knowledge benchmark` | multiple-choice 是否正确 | MMLU |
| `ranking` | 多模型相对胜率 | Elo / Bradley-Terry |

## MATH-500 bridge

`reasoning-from-scratch/ch03/02_math500-verifier-scripts` 已有 verifier。使用方式：

```bash
cd /sgl-workspace/zkx/train/reasoning-from-scratch/ch03/02_math500-verifier-scripts
python evaluate_json.py --help
```

本教程不复制 verifier 逻辑，而是在 `scripts/50_run_math500_eval_bridge.sh` 中调用原仓库脚本。

## MMLU

`reasoning-from-scratch/chF/02_mmlu` 覆盖三种方式：

1. `letter matching`。
2. `logprob scoring`。
3. `teacher forcing`。

服务器训练中建议优先实现 `logprob scoring` 或 `teacher forcing`，因为它们比直接生成字母更稳定。

## LLM-as-a-judge

`LLM-as-a-judge` 适合比较 instruction response，但不要把它当成唯一指标。建议同时保存：

1. prompt。
2. model answer。
3. reference answer。
4. judge rationale。
5. judge score。

## Evaluation cadence

教学 run：

```text
every checkpoint -> small eval
every major milestone -> full eval
```

正式 run：

1. 每 N steps 跑 validation loss。
2. 每个保存的 checkpoint 跑 small benchmark。
3. 只对候选 checkpoint 跑 full MATH-500/MMLU/judge。

## Avoid benchmark leakage

不要把 MATH-500、MMLU test set、leaderboard eval set 混进 training/SFT/distillation 数据。reasoning 模型尤其容易因为数据污染产生虚假提升。
