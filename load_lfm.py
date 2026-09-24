"""Serve LFM2.5 GGUFs over HTTP via llama.cpp's llama-server, so models stay
loaded between requests.

Start:  .venv/bin/python load_lfm.py [--port 8081]

POST /chat    {"messages": [{"role": "user", "content": "..."}], "model": "dspark"}
              also accepts OpenAI fields: temperature, max_tokens, ...
GET  /health  -> loaded models

Each model alias spawns one `llama-server` subprocess on an internal port
(8091+) on first use; this file proxies /chat to it as OpenAI
/v1/chat/completions. `dspark` (default) runs the instruct model with the
DSpark draft for speculative decoding; `instruct` runs it plain.
"""
import argparse
import atexit
import json
import os
import signal
import subprocess
import threading
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "models")

# locally built llama-server (git master, has 'dflash'/DSpark support); the
# distro package (b6153) can't load the DSpark draft at all
LLAMA_SERVER = os.path.join(HERE, "bin", "llama-server")
if not os.path.exists(LLAMA_SERVER):
    LLAMA_SERVER = "llama-server"

INSTRUCT = os.path.join(BASE, "LFM2.5-1.2B-Instruct-Q4_K_M.gguf")
DSPARK = os.path.join(BASE, "LFM2.5-1.2B-Instruct-DSpark-Q4_K_M.gguf")

# dspark is not a standalone model - it is a speculative-decoding draft for the
# instruct model, so its config loads both
MODELS = {
    "instruct": ["-m", INSTRUCT, "--jinja"],
    "dspark": ["-m", INSTRUCT, "-md", DSPARK,
               "--spec-type", "draft-dspark", "--spec-draft-n-max", "7",
               "-fa", "on", "--jinja"],
}
DEFAULT_MODEL = "dspark"
INTERNAL_PORT_BASE = 8091  # each model's llama-server gets 8091, 8092, ...
CTX_SIZE = 4096
LOAD_TIMEOUT = 180  # seconds to wait for a llama-server to finish loading

_backends = {}  # model name -> {"proc": Popen, "port": int}
_lock = threading.Lock()


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _wait_ready(port):
    """Poll llama-server /health until it answers 200 (returns 503 while loading)."""
    t0 = time.time()
    while time.time() - t0 < LOAD_TIMEOUT:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except urllib.error.HTTPError as e:
            if e.code != 503:
                raise
        except OSError:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"llama-server on :{port} not ready after {LOAD_TIMEOUT}s")


def backend(name):
    """Return (or spawn) the llama-server subprocess for a model alias."""
    args = MODELS.get(name) or ["-m", name, "--jinja"]  # alias or .gguf path
    with _lock:
        if name in _backends and _backends[name]["proc"].poll() is None:
            return _backends[name]
        port = INTERNAL_PORT_BASE + len(_backends)
        _log(f"spawning llama-server for {name} on :{port} ...")
        t0 = time.time()
        proc = subprocess.Popen(
            [LLAMA_SERVER, *args, "--host", "127.0.0.1", "--port", str(port),
             "-c", str(CTX_SIZE), "-ngl", "0", "--log-disable"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _wait_ready(port)
        _log(f"  ready in {time.time() - t0:.1f}s")
        _backends[name] = {"proc": proc, "port": port}
        return _backends[name]


def _kill_all():
    for name, b in _backends.items():
        if b["proc"].poll() is None:
            b["proc"].terminate()


atexit.register(_kill_all)
signal.signal(signal.SIGTERM, lambda *_: (_kill_all(), exit(0)))


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
            self._send(200, {"status": "ok", "loaded": list(_backends), "models": MODELS})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/chat", "/v1/chat/completions"):
            _log(f"POST {self.path} -> 404")
            self._send(404, {"error": "not found"})
            return
        try:
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except json.JSONDecodeError as e:
            _log(f"POST /chat -> 400 bad json: {e}")
            self._send(400, {"error": f"bad json: {e}"})
            return
        model = payload.get("model", DEFAULT_MODEL)
        msgs = payload.get("messages") or ([{"role": "user", "content": payload["message"]}]
                                           if "message" in payload else None)
        if not msgs:
            _log("POST /chat -> 400 no messages")
            self._send(400, {"error": "need {messages: [{role, content}...], model?}"})
            return
        last = msgs[-1].get("content", "")
        _log(f"POST /chat model={model} msgs={len(msgs)} last={last[:120]!r}"
             f"{'...' if len(last) > 120 else ''}")
        try:
            b = backend(model)
            req = dict(payload, model=model, messages=msgs)
            data = json.dumps(req).encode()
            r = urllib.request.Request(
                f"http://127.0.0.1:{b['port']}/v1/chat/completions", data=data,
                headers={"Content-Type": "application/json"})
            t0 = time.time()
            with urllib.request.urlopen(r, timeout=300) as resp:
                result = json.loads(resp.read())
            ms = round((time.time() - t0) * 1000)
            usage = result.get("usage", {})
            reply = result["choices"][0]["message"]["content"]
            _log(f"  -> 200 {ms}ms {usage.get('completion_tokens', '?')}tok reply={reply[:120]!r}")
            self._send(200, result)
        except (KeyError, ValueError) as e:
            _log(f"  -> 400 {e}")
            self._send(400, {"error": str(e)})
        except Exception as e:
            _log(f"  -> 500 {type(e).__name__}: {e}")
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        _log(f"{self.address_string()} {fmt % args}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--model", default=DEFAULT_MODEL, help="model alias to preload")
    args = p.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)  # bind first: fail fast if port is taken
    backend(args.model)  # preload so a bad model fails at startup
    print(f"serving on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
