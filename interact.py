"""Basic terminal chat with the LFM server (load_lfm.py).

Start the server first:  .venv/bin/python load_lfm.py
Then:                    .venv/bin/python interact.py

Commands:  /reset   clear history
           /model <instruct|dspark>   switch model
           quit / exit   leave
"""
import httpx

SERVER = "http://127.0.0.1:8081"
MODEL = "dspark"
SYSTEM = "You are a helpful assistant."

history = [{"role": "system", "content": SYSTEM}]
model = MODEL
print(f"chatting with {model} on {SERVER} — 'quit' to leave, /reset to clear")
while True:
    try:
        msg = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not msg:
        continue
    if msg in ("quit", "exit"):
        break
    if msg == "/reset":
        history = [{"role": "system", "content": SYSTEM}]
        print("(history cleared)")
        continue
    if msg.startswith("/model "):
        model = msg.split(None, 1)[1]
        print(f"(model -> {model})")
        continue
    history.append({"role": "user", "content": msg})
    try:
        resp = httpx.post(f"{SERVER}/chat", timeout=300,
                          json={"messages": history, "model": model})
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"error: {e}")
        history.pop()
        continue
    history.append({"role": "assistant", "content": reply})
    print(f"lfm> {reply}")
