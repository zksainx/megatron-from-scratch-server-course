# Data Artifacts

Do not commit generated datasets or Megatron runtime outputs.

Rebuild the tutorial JSONL files from the two source repositories:

```bash
python scripts/10_prepare_from_scratch_datasets.py \
  --train-root /sgl-workspace/zkx/train \
  --output-dir data/from_scratch
```

Then build Megatron `.bin/.idx` files:

```bash
bash scripts/11_preprocess_gpt_data.sh \
  data/from_scratch/pretrain_the_verdict.jsonl \
  runs/data/the_verdict
```

The source datasets live in `LLMs-from-scratch` and `reasoning-from-scratch`; this course repository only stores scripts and documentation that derive training artifacts from those local copies.
