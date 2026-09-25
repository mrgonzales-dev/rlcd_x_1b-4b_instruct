import json
import time

import httpx

from people_data import DATA

# ── SERVERS ───────────────────────────────────────────────────────────────
# Requires: .venv/bin/python load_laya.py   (laya on :8000)
#           .venv/bin/python load_lfm.py    (LFM2.5 on :8081)
LAYA = "http://127.0.0.1:8000"
LFM = "http://127.0.0.1:8081"
LAYA_MODEL = "english"
LFM_MODEL = "dspark"
TOP_K = 1  # only the top record goes to lfm - extra records stain the reply

# ── CONTEXT LIMITER ───────────────────────────────────────────────────────
# llama-server runs with -c 32768 (see load_lfm.py). No tokenizer here, so we
# estimate ~4 chars/token and drop oldest turns when over budget.
MAX_CTX_TOKENS = 32768
SYSTEM = {"role": "system", "content": "You are a helpful assistant. When given database records, answer from them."}

def est_tokens(messages):
    return sum(len(m["content"]) // 4 + 4 for m in messages)

def fit(messages, budget=MAX_CTX_TOKENS):
    """Keep the system message + as many recent turns as fit in the budget."""
    head = messages[:1] if messages and messages[0]["role"] == "system" else []
    rest = messages[len(head):]
    while rest and est_tokens(head + rest) > budget:
        rest.pop(0)
    return head + rest

# ── COMMANDS: the user picks the tool, no routing model needed ────────────
COMMANDS = {"/laya": "laya_search"}

# ── THE TOOL: laya ranks every record by "can this entry answer it?" ──────
def describe(key, value):
    """Compact option label per record, kept under laya's head budget."""
    parts = [value.get("full_name", key.replace("_", " "))]
    for field in ("relationships", "occupation"):
        v = value.get(field)
        if isinstance(v, dict):
            v = ", ".join(f"{k} {x}" for k, x in v.items())
        if v:
            parts.append(str(v))
    return ", ".join(parts)[:90]

def laya_search(query):
    """One laya choice call over all records -> ranked (prob, key, record)."""
    r = httpx.post(f"{LAYA}/evaluate", timeout=120, json={
        "state": {"question": query},
        "questions": {"pick": {
            "type": "choice",
            "instructions": "Which entry's data can answer the question?",
            "criteria": {k: describe(k, v) for k, v in DATA.items()}}},
        "model": LAYA_MODEL})
    result = r.json()
    if r.status_code != 200:
        raise SystemExit(f"laya returned {r.status_code}: {result.get('error')}")
    probs = result["answers"]["pick"]["probabilities"]
    return sorted(((probs.get(k, 0.0), k, v) for k, v in DATA.items()), reverse=True)

# ── LFM: the replier ──────────────────────────────────────────────────────
def chat(messages, temperature=0.1):
    messages = fit(messages)
    r = httpx.post(f"{LFM}/chat", json={
        "model": LFM_MODEL, "messages": messages, "temperature": temperature},
        timeout=300)
    result = r.json()
    if r.status_code != 200:
        raise SystemExit(f"lfm returned {r.status_code}: {result.get('error')}")
    return result["choices"][0]["message"]["content"].strip()

def answer(query, ranked, history):
    """LFM answers using the top-ranked records + chat history (for follow-ups).
    The record context is transient - only the clean query stays in history."""
    context = "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for _, k, v in ranked)
    return chat([SYSTEM] + history[:-1]
                + [{"role": "user", "content": f"Database records:\n{context}\n\n{query}"}])

# ── RUN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("commands: /laya <query> searches the people database; anything else is chat (q to quit)")
    history = []
    while True:
        try:
            msg = input("\nyou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not msg or msg.lower() == "q":
            break

        t0 = time.time()
        cmd = next((c for c in COMMANDS if msg.startswith(c)), None)
        if not cmd:
            history.append({"role": "user", "content": msg})
            reply = chat([SYSTEM] + history)
            history.append({"role": "assistant", "content": reply})
            print(f"lfm: {reply}  [{time.time() - t0:.1f}s, ~{est_tokens([SYSTEM] + history)} tok]")
            continue

        # 1. build the tool call directly - the arg IS the text after /laya
        query = msg[len(cmd):].strip()
        call = {"name": COMMANDS[cmd], "arguments": {"query": query}}
        print(f"tool call: {json.dumps(call)}")

        # 2. laya IS the tool - ranks the records
        ranked = laya_search(query)
        for p, k, v in ranked[:TOP_K]:
            print(f"  {p:.2f}  {k}: {v.get('full_name', '')}")

        # 3. lfm replies from what laya found (with history for follow-ups)
        history.append({"role": "user", "content": query})
        reply = answer(query, ranked[:TOP_K], history)
        history.append({"role": "assistant", "content": reply})
        print(f"lfm: {reply}  [{time.time() - t0:.1f}s, ~{est_tokens([SYSTEM] + history)} tok]")
