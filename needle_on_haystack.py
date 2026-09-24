"""Needle-in-a-haystack: pick which enclosure of a JSON dataset answers a question.

Laya gets one "choice" call per query — the query is the state, every top-level
key is an option — and returns a probability per enclosure. The top-ranked
enclosures (the needles) go to LFM for a natural-language answer.

Run:  .venv/bin/python needle_on_haystack.py
      (requires load_laya.py and load_lfm.py running)
"""
import json
import resource
import sys
import time

import httpx

from people_data import DATA

# ── LAYA ──────────────────────────────────────────────────────────────────
# Scores each field for relevance. Requires load_laya.py running:
#   .venv/bin/python load_laya.py
LAYA_SERVER = "http://127.0.0.1:8000"
LAYA_MODEL = "english"

# ── LFM ───────────────────────────────────────────────────────────────────
# Answers each query from the top-scored fields. Requires load_lfm.py running:
#   .venv/bin/python load_lfm.py
LFM_SERVER = "http://127.0.0.1:8081"
LFM_MODEL = "dspark"

# Generation tuning — passed through to llama.cpp's OpenAI-compatible API.
# temperature 0.0 + fixed seed = deterministic answers for testing.
LFM_SYSTEM = "Answer the user's question with the data given."
LFM_CONFIG = {
    "temperature": 0.0,      # 0 = always pick the most likely token (deterministic); higher = more random
    "top_p": 0.9,            # only sample from the smallest token set covering 90% probability mass
    "top_k": 40,             # only sample from the 40 most likely tokens
    "max_tokens": 256,       # hard cap on answer length
    "repeat_penalty": 1.1,   # >1 discourages repeating the same words; 1.0 = off
    "seed": 42,              # fixed RNG seed so runs are reproducible
}

# ── OLLAMA (optional backend) ─────────────────────────────────────────────
# Run with --ollama to answer via a local Ollama model instead of load_lfm.py:
#   ollama pull minicpm-v && ollama serve
#   .venv/bin/python needle_on_haystack.py --ollama
USE_OLLAMA = "--ollama" in sys.argv
OLLAMA_SERVER = "http://127.0.0.1:11434"
OLLAMA_MODEL = "minicpm-v"
BACKEND = "ollama" if USE_OLLAMA else "lfm"

# ── HAYSTACK ──────────────────────────────────────────────────────────────
# DATA is imported from people_data.py — 50 Filipino celebrities, one key each.

# ── SCAN ──────────────────────────────────────────────────────────────────
# One choice call: the query is the state, every enclosure is an option.
# The model's head budget is small (~192 tokens for all options), so each
# option description must be short: name + relationships + occupation.
def describe(key, value):
    """Compact option label for an enclosure, kept under the head budget."""
    if not isinstance(value, dict):
        return str(value)[:80]
    parts = [value.get("full_name", key.replace("_", " "))]
    for field_name in ("relationships", "occupation"):
        field_value = value.get(field_name)
        if isinstance(field_value, dict):
            field_value = ", ".join(f"{k} {v}" for k, v in field_value.items())
        if field_value:
            parts.append(str(field_value))
    return ", ".join(parts)[:90]


def summarize(query, ranked_fields):
    """Ask the answer model (LFM, or Ollama with --ollama) using ranked fields only."""
    context = "\n".join(f"{key}: {json.dumps(value, ensure_ascii=False)}"
                        for _, key, value in ranked_fields)
    messages = [
        {"role": "system", "content": LFM_SYSTEM},
        {"role": "user", "content": f"{context}\n\n{query}"},
    ]
    if USE_OLLAMA:
        response = httpx.post(f"{OLLAMA_SERVER}/api/chat", timeout=120, json={
            "model": OLLAMA_MODEL,
            "stream": False,
            "messages": messages,
            "options": {
                "temperature": LFM_CONFIG["temperature"],
                "top_p": LFM_CONFIG["top_p"],
                "top_k": LFM_CONFIG["top_k"],
                "num_predict": LFM_CONFIG["max_tokens"],
                "repeat_penalty": LFM_CONFIG["repeat_penalty"],
                "seed": LFM_CONFIG["seed"],
            },
        })
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
    response = httpx.post(f"{LFM_SERVER}/chat", timeout=120, json={
        "model": LFM_MODEL,
        "messages": messages,
        **LFM_CONFIG,
    })
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def laya_rank(query, fields, elapsed):
    """One laya choice call: the query is the state, enclosures are the options.
    Returns (probability, key, value) tuples ranked highest first."""
    print(f"[{elapsed():5.1f}s] laya: choosing from {len(fields)} fields")
    response = httpx.post(f"{LAYA_SERVER}/evaluate", timeout=120, json={
        "state": {"question": query},
        "questions": {"pick": {
            "type": "choice",
            "instructions": "Which entry's data can answer the question?",
            "criteria": {key: describe(key, value) for key, value in fields},
        }},
        "model": LAYA_MODEL})
    result = response.json()
    if response.status_code != 200:
        raise SystemExit(f"{LAYA_SERVER} returned {response.status_code}: {result.get('error')}")
    probabilities = result["answers"]["pick"]["probabilities"]
    scored = sorted(((probabilities.get(key, 0.0), key, value) for key, value in fields),
                    reverse=True)
    for probability, key, _ in scored:
        print(f"[{elapsed():5.1f}s]   {key} -> {probability:.2f}")
    return scored


def run_query(query, fields, timings):
    """Rank the enclosures, summarize the top-scored data."""
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    print(f"\n{'=' * 64}\n{query}")
    scored = laya_rank(query, fields, elapsed)

    print(f"[{elapsed():5.1f}s] ranked:")
    for score, key, value in scored:
        name = value.get("full_name", "") if isinstance(value, dict) else ""
        print(f"  {score:5.2f}  {key}: {name or json.dumps(value, ensure_ascii=False)[:60]}")

    scan_time = time.time() - start_time
    best_score, best_key, best_value = scored[0]
    print(f"  -> needle: {best_key} = {json.dumps(best_value, ensure_ascii=False)[:120]}  [{scan_time:.1f}s]")

    top_keys = [key for _, key, _ in scored[:1]]
    print(f"[{elapsed():5.1f}s] {BACKEND}: summarizing ranked fields: {', '.join(top_keys)}")
    answer = summarize(query, scored[:1])
    lfm_time = time.time() - start_time - scan_time
    print(f"[{elapsed():5.1f}s] {BACKEND}: done")
    print(f"  -> {BACKEND}: {answer}")
    timings[query] = time.time() - start_time
    return scan_time, lfm_time


if __name__ == "__main__":
    total_start = time.time()
    timings = {}
    laya_time = 0.0
    lfm_time = 0.0
    fields = list(DATA.items())

    while True:
        try:
            query = input("\nquestion (q to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() == "q":
            break
        scan_time, answer_time = run_query(query, fields, timings)
        laya_time += scan_time
        lfm_time += answer_time

    # ── METRICS ───────────────────────────────────────────────────────────
    # RAM is this client process only — the models live in the load_*.py servers.
    peak_ram_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"\n{'=' * 64}\nquestion runtime")
    for query, elapsed in timings.items():
        print(f"  {elapsed:5.1f}s  {query}")

    print(f"\n{'=' * 64}\nmetric breakdown")
    print(f"  {laya_time:5.1f}s  laya total")
    print(f"  {lfm_time:5.1f}s  {BACKEND} total")
    print(f"  {time.time() - total_start:5.1f}s  total")
    print(f"  peak ram: {peak_ram_mb:.0f} MB")
