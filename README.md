# Shakespeare LLM
Character-level transformer trained on the works of William Shakespeare.

Prereqs
- Python 3.10+ (this repo has a `.venv` in the project root used in examples)
- Node.js + npm (for the frontend)
- Optional: a GPU or Apple Silicon (MPS) for faster sampling/training

Quick CLI (train / sample)

1) Activate the repo venv (from repository root):

```
source .venv/bin/activate
```

2) Install Python runtime deps if you haven't:

```
python -m pip install torch einops requests numpy
```

3) Train (example):

```
python3 train.py --num-epochs 5 --batch-size 64 --block-size 128 --dim 256 --num-heads 4 --num-layers 6
```

Checkpoints
- Checkpoints are saved to `checkpoints/epoch_XXX.pt`.
- The CLI and example server default to picking the best available device (MPS > CUDA > CPU); you may force a device with `--device cuda|mps|cpu`.

Sampling from a checkpoint (CLI):

```
python3 sample.py checkpoints/epoch_001.pt --prefix "ROMEO:" --max-new-tokens 300 --temperature 0.8
```

Frontend demo (quick)

1) Frontend dev server (in `front-end/`):

```
cd front-end
npm install
VITE_API_BASE=http://localhost:8000 npm run dev
```

2) Example backend (optional) — quick server that calls the sampling code when a checkpoint is provided.

- From the project root use the repo venv python to run the example server and point it at a checkpoint:

```
CHECKPOINT_PATH=$(pwd)/checkpoints/epoch_001.pt .venv/bin/python3 front-end/example_server.py
```

- If the model or dependencies (torch) are missing the server will fall back to echo mode; logs show whether the checkpoint loaded successfully.

Single-service (production-style)

Build the frontend and serve it from the same Python server (same origin for API + UI).

1) Build the frontend:

```
cd front-end
npm install
npm run build
```

2) Install backend deps (from repo root):

```
python -m pip install -r requirements.txt
```

3) Run the server (from repo root):

```
CHECKPOINT_PATH=$(pwd)/checkpoints/epoch_001.pt PORT=8000 .venv/bin/python3 front-end/example_server.py
```

4) Open http://localhost:8000

Notes:
- In production builds, the frontend defaults to same-origin `/api/generate`. If you need a different API host, set `VITE_API_BASE` at build time.
- The server will serve `front-end/dist` automatically if it exists. If you only see the server HTML message, re-run the frontend build.

Notes
- The frontend expects `POST /api/generate` with JSON { prompt: string, max_new_tokens?: number } and returns { generated_text }.
- For development the example server enables CORS and a simple GET helper at `/api/generate?prompt=...` for quick browser tests.
