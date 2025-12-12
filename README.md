# Shakespeare LLM

Character-level transformer trained on Tiny Shakespeare. Data downloads automatically and checkpoints are saved every epoch.

## Setup
Using `uv` (recommended):
```bash
uv venv
source .venv/bin/activate
uv pip install torch einops requests numpy
```

## Quickstart: Train
```bash
python3 train.py \
  --num-epochs 5 \
  --batch-size 64 \
  --block-size 128 \
  --dim 256 \
  --num-heads 4 \
  --num-layers 6
```
- Checkpoints land in `checkpoints/epoch_XXX.pt` (includes optimizer state).
- Device auto-picks mps > cuda > cpu; override with `--device cpu|cuda|mps`.

## Quickstart: Sample
```bash
python3 sample.py checkpoints/epoch_005.pt \
  --prefix "ROMEO:" \
  --max-new-tokens 300 \
  --temperature 0.8 \
  --top-p 0.9
```
- Tweak `--top-k`, `--temperature`, or `--block-size` to change style/entropy.
- Replace `--prefix` to steer the opening line.
