"""
RoboPacer web server.
Run: python engine/server.py
Then open: http://localhost:5000
"""
from flask import Flask, Response, request, jsonify, send_file
import threading, json, time, sys, os, traceback
from pathlib import Path

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
ENGINE_DIR = Path(__file__).parent

sys.path.insert(0, str(ENGINE_DIR))

app = Flask(__name__)

# ── Job state ─────────────────────────────────────────────────────────────────
_lock    = threading.Lock()
_running = False
_events  = []   # list of SSE event dicts, appended during a job


def _push(event: dict):
    _events.append(event)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def serve_index():
    return send_file(ROOT / "index.html")


@app.route("/api/status")
def api_status():
    MODELS_DIR.mkdir(exist_ok=True)
    exts = {".hef", ".onnx", ".har", ".pth"}
    mods = sorted(f.name for f in MODELS_DIR.iterdir()
                  if f.is_file() and f.suffix in exts)
    return jsonify({"ok": True, "running": _running, "models": mods})


@app.route("/api/train", methods=["POST"])
def api_train():
    global _running, _events
    with _lock:
        if _running:
            return jsonify({"error": "A job is already running."}), 409
        config   = request.json or {}
        _running = True
        _events  = []

    def run():
        global _running
        try:
            import train
            train.run(config, _push)
        except Exception as e:
            _push({"type": "log", "text": traceback.format_exc(), "level": "error"})
        finally:
            _push({"type": "done"})
            _running = False

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/stream")
def api_stream():
    """Server-Sent Events — streams events from the current job."""
    def generate():
        last       = 0
        sent_done  = False
        # Wait up to 60 s for the job to produce at least one event
        waited = 0
        while not _events and waited < 60:
            time.sleep(0.2)
            waited += 0.2
        while True:
            chunk = _events[last:]
            for ev in chunk:
                yield f"data: {json.dumps(ev)}\n\n"
                if ev.get("type") == "done":
                    sent_done = True
            last += len(chunk)
            if sent_done:
                break
            time.sleep(0.1)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":       "no-cache",
            "X-Accel-Buffering":   "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.after_request
def cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    MODELS_DIR.mkdir(exist_ok=True)
    print("=" * 55)
    print("  RoboPacer Studio  →  http://localhost:5000")
    print("  Press Ctrl+C to stop.")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
