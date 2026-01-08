"""Example dev server for the frontend. Run this to provide POST /api/generate.

This server will echo prompts by default. If you want real samples, update the
`generate_from_model` function to call into your model sampling code.
"""
from flask import Flask, request, jsonify, make_response, send_from_directory
import time
import os
from typing import Optional


# Repository + frontend paths (used for serving the built frontend).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FRONTEND_DIST = os.path.join(REPO_ROOT, 'front-end', 'dist')

# Try to import the sampling utilities from the repo; if unavailable, we'll fall back to echo.
MODEL_AVAILABLE = False
model = None
stoi = None
itos = None
model_device = None

try:
    # import functions from sample.py (in the repository root)
    # make sure Python path includes repo root
    if REPO_ROOT not in os.sys.path:
        os.sys.path.insert(0, REPO_ROOT)

    from sample import load_checkpoint, sample as sample_from_checkpoint, select_device
    import torch
    MODEL_AVAILABLE = True
except Exception as e:
    print('Model imports not available; falling back to echo. Import error:', e)
    MODEL_AVAILABLE = False

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    # Allow the Vite dev server (or other origins) to call this API during development
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response


def generate_from_model(prompt: str, max_new_tokens: int = 200) -> str:
    # If a model is loaded, call the sampling function; otherwise echo for convenience.
    time.sleep(0.1)
    if MODEL_AVAILABLE and model is not None:
        try:
            # sample_from_checkpoint will move model to device itself, but ensure device is set
            generated = sample_from_checkpoint(
                model,
                stoi,
                itos,
                prefix=prompt,
                max_new_tokens=max_new_tokens,
                device=model_device,
            )
            # If the model returned the full sequence including the prompt, strip the prompt
            if isinstance(generated, str) and generated.startswith(prompt):
                return generated[len(prompt):].lstrip()
            return generated
        except Exception as ex:
            print('Error during model sampling, falling back to echo:', ex)
            return f"[error generating] {ex}"

    return f"[echo] {prompt}"


@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.get_json() or {}
    prompt = data.get('prompt', '')
    max_new_tokens = int(data.get('max_new_tokens', 200))
    text = generate_from_model(prompt, max_new_tokens)
    return jsonify({ 'generated_text': text })



@app.route('/api/generate', methods=['GET'])
def api_generate_get():
    # Provide a helpful GET interface for quick testing in a browser.
    prompt = request.args.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'This endpoint accepts POST for programmatic use. For quick testing provide ?prompt=...'}), 200
    text = generate_from_model(prompt, int(request.args.get('max_new_tokens', 200)))
    return jsonify({'generated_text': text})


@app.route('/')
@app.route('/<path:path>')
def serve_frontend(path: str = ''):
    # Serve the built frontend if present; otherwise show a helpful message.
    if not os.path.isdir(FRONTEND_DIST):
        html = '''
        <html><body>
          <h3>Shakespeare LLM Example Server</h3>
          <p>This server provides <code>/api/generate</code> for POST requests from the frontend.</p>
          <p>To build the frontend, run <code>npm install</code> and <code>npm run build</code> in <code>front-end</code>.</p>
          <p>To run the frontend dev server instead, run: <code>npm run dev</code> in <code>front-end</code>.</p>
          <p>To quickly test generation from a browser, try: <a href="/api/generate?prompt=ROMEO:%20Hello">/api/generate?prompt=ROMEO:%20Hello</a></p>
        </body></html>
        '''
        resp = make_response(html)
        resp.headers['Content-Type'] = 'text/html'
        return resp

    if path and os.path.exists(os.path.join(FRONTEND_DIST, path)):
        return send_from_directory(FRONTEND_DIST, path)
    return send_from_directory(FRONTEND_DIST, 'index.html')


if __name__ == '__main__':
    # Try to load a checkpoint at startup if provided via env var CHECKPOINT_PATH
    checkpoint_path = os.environ.get('CHECKPOINT_PATH') or os.environ.get('CHECKPOINT')
    if MODEL_AVAILABLE and checkpoint_path:
        try:
            model_device = select_device(None)
            print(f'Loading checkpoint {checkpoint_path} onto device {model_device} ...')
            m, s, i, cfg = load_checkpoint(checkpoint_path, device=model_device)
            model = m
            stoi = s
            itos = i
            print('Checkpoint loaded successfully.')
        except Exception as e:
            print('Failed to load checkpoint; continuing with echo fallback. Error:', e)

    port = int(os.environ.get('SERVER_PORT', os.environ.get('PORT', 8000)))
    app.run(host='0.0.0.0', port=port)
