"""Serve laya over HTTP so the model stays loaded between test runs.

Start:  .venv/bin/python load_laya.py [--port 8000]

POST /evaluate  {"state": ..., "questions": {...}, "model": "english"}
GET  /health    -> loaded models
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "laya")
sys.path.insert(0, BASE)
from rl_agent_api import RLAgent

MODELS = {
    "english": BASE,
    "multilingual": BASE + "/multilingual",
    "typed-decisions": BASE + "/typed-decisions",
}
DEFAULT_MODEL = "english"

_agents = {}
_lock = threading.Lock()


def agent(name):
    path = MODELS.get(name, name)  # alias or absolute checkpoint path
    with _lock:
        if path not in _agents:
            print(f"loading {path} ...", flush=True)
            t0 = time.time()
            _agents[path] = RLAgent(path, device="cpu")
            print(f"  loaded in {time.time() - t0:.1f}s", flush=True)
        return _agents[path]


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        _log(f"GET {self.path}")
        if self.path == "/health":
            self._send(200, {"status": "ok", "loaded": list(_agents), "models": MODELS})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/evaluate":
            _log(f"POST {self.path} -> 404")
            self._send(404, {"error": "not found"})
            return
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except json.JSONDecodeError as e:
            _log(f"POST /evaluate -> 400 bad json: {e}")
            self._send(400, {"error": f"bad request, need {{state, questions, model?}}: {e}"})
            return
        model = payload.get("model", DEFAULT_MODEL)
        state = json.dumps(payload.get("state"), ensure_ascii=False)
        _log(f"POST /evaluate model={model} questions={list(payload.get('questions', {}))} "
             f"state={state[:120]}{'...' if len(state) > 120 else ''}")
        try:
            t0 = time.time()
            result = agent(model).system_one(payload["state"], payload["questions"])
            result["elapsed_ms"] = round((time.time() - t0) * 1000)
            summary = {qid: a.get("choice", a.get("score", a.get("noul")))
                       for qid, a in result["answers"].items()}
            _log(f"  -> 200 {result['elapsed_ms']}ms {result['usage']['input_tokens']}tok answers={summary}")
            self._send(200, result)
        except (KeyError, TypeError) as e:
            _log(f"  -> 400 {e}")
            self._send(400, {"error": f"bad request, need {{state, questions, model?}}: {e}"})
        except Exception as e:
            _log(f"  -> 500 {type(e).__name__}: {e}")
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        _log(f"{self.address_string()} {fmt % args}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default=DEFAULT_MODEL, help="checkpoint to preload")
    args = p.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)  # bind first: fail fast if port is taken
    agent(args.model)  # preload so a bad checkpoint fails at startup
    print(f"serving on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
