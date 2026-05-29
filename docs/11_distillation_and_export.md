# Distillation And Export

这章覆盖 `reasoning-from-scratch` Ch 8：用 teacher model 生成 reasoning data，再训练 student model。

## Distillation pipeline

```text
prompt dataset
  -> teacher generation
  -> filtering / verifier scoring
  -> messages JSONL
  -> Megatron SFT
  -> evaluation
```

## Teacher generation

可以使用：

1. local model server。
2. Ollama。
3. hosted API。
4. another Megatron checkpoint。

输出建议保存为：

```json
{"prompt":"...","teacher":"...","answer":"...","score":1.0,"metadata":{"teacher_model":"..."}}
```

然后转换成 SFT:

```json
{"messages":[{"role":"system","content":"Solve the problem and put the final answer in \\boxed{}."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

## Filtering

只训练正确 teacher output 通常更稳：

1. 用 verifier 判断 final answer。
2. 去掉空回答、超长回答、格式错误回答。
3. 去掉重复 prompt。
4. 保留 failure case 用于 evaluation，不要混入 train。

## Export

Megatron checkpoint 不一定能直接用 Hugging Face 加载。常见选择：

1. 继续在 Megatron 中 inference/eval。
2. 用 Megatron checkpoint conversion 工具转 HF。
3. 写 wrapper，让 HF 风格代码调用 Megatron 权重。

`reasoning-from-scratch/ch08/06_use_via_huggingface` 展示了 wrapper/export 思路。Megatron 侧可参考：

```bash
python /sgl-workspace/zkx/train/Megatron-LM/tools/checkpoint/convert.py --help
```

## Student selection

distillation 后不要只看训练 loss。至少比较：

1. base model。
2. SFT model。
3. distilled model。
4. distilled + inference-time scaling。
5. distilled + RLVR。
