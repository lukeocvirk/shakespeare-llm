# Shakespeare LLM Frontend

This is a small React + Vite frontend to demo the Tiny Shakespeare character-level model in this repo.

Run locally (from this folder):

1. Install dependencies

   npm install

2. Start dev server

   npm run dev

Environment
- By default the frontend will POST to http://localhost:8000/api/generate. Set VITE_API_BASE to change the API base URL (for example: VITE_API_BASE=http://localhost:5000 npm run dev).

Backend
- The backend should implement POST /api/generate that accepts JSON { prompt: string, max_new_tokens?: number } and responds with JSON containing generated text (e.g. { generated_text: "..." }). The repo's Python code (train.py/sample.py) includes sampling code you can adapt to serve this endpoint.
