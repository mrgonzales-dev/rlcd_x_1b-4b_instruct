import json
import httpx

# ── SERVER ────────────────────────────────────────────────────────────────
# Requires load_laya.py running:  .venv/bin/python load_laya.py
SERVER = "http://127.0.0.1:8000"
MODEL = "english"

# ── TOOLS the llm could call ───────────────────────────────────────────────
TOOLS = {
    "get_weather": "current weather or forecast for a location",
    "web_search": "look up facts, news, or information on the internet",
    "calculator": "evaluate a math expression",
    "send_email": "compose and send an email",
}
CHOICE_CRITERIA = {"none": "answer directly, no tool needed", **TOOLS}

# ── CASES ─────────────────────────────────────────────────────────────────
CASES = [
    # single tool
    {"state": "Hey, what's the weather like in Tokyo right now?", "expect": {"get_weather"}},
    {"state": "What are today's top headlines?", "expect": {"web_search"}},
    {"state": "What is 128 * 46 + 17?", "expect": {"calculator"}},
    {"state": "Send an email to john@acme.com saying the meeting moved to 3pm.", "expect": {"send_email"}},
    # multiple tools
    {"state": "Check the weather in Tokyo and email it to john@acme.com.", "expect": {"get_weather", "send_email"}},
    {"state": "Search for the cheapest flights to Lisbon and email the result to ana@acme.com.",
     "expect": {"web_search", "send_email"}},
    {"state": "What's a 15% tip on $84.50, and will it rain in London tomorrow?",
     "expect": {"calculator", "get_weather"}},
    {"state": "Find the latest Fed rate news, check if it's sunny in Miami, and email both to me.",
     "expect": {"web_search", "get_weather", "send_email"}},
    # no tool
    {"state": "What is the capital of France?", "expect": set()},
    {"state": "Explain how photosynthesis works.", "expect": set()},
    {"state": "Write a haiku about autumn.", "expect": set()},
    {"state": "Hi! How are you today?", "expect": set()},
]

# ── EXPERIMENT 1: noul phrasing variants ──────────────────────────────────
# Same question type, different framing. {t}=tool name, {d}=description.
VARIANTS = {
    "help":     "Should the assistant call the {t} tool ({d}) to help answer this request?",
    "part":     "Is there any part of this request that needs the {t} tool ({d})?",
    "required": "Must the assistant call the {t} tool ({d}) to fully satisfy this request?",
    "useful":   "Would the {t} tool ({d}) be useful for any part of this request?",
}

# ── EXPERIMENT 2: aggregation rules over per-tool probabilities ───────────
def thr(p, t):
    return {k for k, v in p.items() if v > t}

def relmax(p, alpha=0.5):
    mx = max(p.values())
    return {k for k, v in p.items() if v >= alpha * mx} if mx > 0.5 else set()

def gap(p):
    s = sorted(p.values(), reverse=True)
    if s[0] <= 0.5:
        return set()
    i = max(range(len(s) - 1), key=lambda j: s[j] - s[j + 1])  # largest drop
    return {k for k, v in p.items() if v >= s[i]}

def sumcount(p):  # expected count = sum of probs; take that many by rank
    return set(sorted(p, key=p.get, reverse=True)[:round(sum(p.values()))])

PROB_RULES = {
    "thr>.2": lambda p: thr(p, 0.2),
    "thr>.3": lambda p: thr(p, 0.3),
    "thr>.5": lambda p: thr(p, 0.5),
    "thr>.7": lambda p: thr(p, 0.7),
    "relmax": relmax,
    "gap":    gap,
    "sumcnt": sumcount,
}

# ── METRICS ───────────────────────────────────────────────────────────────
def score(preds, expects):
    tp = sum(len(p & e) for p, e in zip(preds, expects))
    fp = sum(len(p - e) for p, e in zip(preds, expects))
    fn = sum(len(e - p) for p, e in zip(preds, expects))
    exact = sum(p == e for p, e in zip(preds, expects))
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return exact, P, R, (2 * P * R / (P + R) if P + R else 0.0)

# ── RUN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # fetch all answers once: choice + each variant's per-tool nouls
    data = []
    for i, case in enumerate(CASES):
        entry = {"expect": case["expect"], "probs": {}}
        r = httpx.post(f"{SERVER}/evaluate", json={
            "state": case["state"],
            "questions": {"which": {"type": "choice",
                                    "instructions": "Which tool should be called to answer this request?",
                                    "criteria": CHOICE_CRITERIA}},
            "model": MODEL}, timeout=120).json()
        entry["choice"] = r["answers"]["which"]["probabilities"]
        for v, ins in VARIANTS.items():
            r = httpx.post(f"{SERVER}/evaluate", json={
                "state": case["state"],
                "questions": {f"call_{t}": {"type": "noul", "instructions": ins.format(t=t, d=d)}
                              for t, d in TOOLS.items()},
                "model": MODEL}, timeout=120).json()
            entry["probs"][v] = {t: r["answers"][f"call_{t}"]["noul"] for t in TOOLS}
        data.append(entry)
        print(f"[{i}] {case['state'][:60]}  expect={sorted(case['expect']) or '[]'}")
        print("    choice: " + json.dumps({k: round(v, 2) for k, v in entry["choice"].items()}))
        for v in VARIANTS:
            print(f"    {v:<9} " + " ".join(f"{t}={entry['probs'][v][t]:.2f}" for t in TOOLS))

    # evaluate every (variant, rule) + cross-variant rules
    expects = [d["expect"] for d in data]
    rows = []
    for v in VARIANTS:
        for rn, rule in PROB_RULES.items():
            rows.append((f"{v}:{rn}", score([rule(d["probs"][v]) for d in data], expects)))
    # single choice (argmax over labels incl. none)
    rows.append(("choice", score([{max(d["choice"], key=d["choice"].get)} - {"none"}
                                  for d in data], expects)))
    # cross-variant combinations
    for rn, rule in {
        "mean>.5":  lambda d: thr({t: sum(d["probs"][v][t] for v in VARIANTS) / len(VARIANTS)
                                    for t in TOOLS}, 0.5),
        "union.5":  lambda d: set().union(*[thr(d["probs"][v], 0.5) for v in VARIANTS]),
        "vote>=2":  lambda d: {t for t in TOOLS
                               if sum(d["probs"][v][t] > 0.5 for v in VARIANTS) >= 2},
        "choice+union": lambda d: ({max(d["choice"], key=d["choice"].get)} - {"none"}) |
                                  set().union(*[thr(d["probs"][v], 0.5) for v in VARIANTS]),
    }.items():
        rows.append((f"all:{rn}", score([rule(d) for d in data], expects)))

    # leaderboard
    print(f"\n{'=' * 78}\n{'config':<20} {'exact':>7} {'prec':>6} {'rec':>6} {'f1':>6}")
    rows.sort(key=lambda r: (-r[1][0], -r[1][3]))
    for name, (exact, P, R, F1) in rows:
        print(f"{name:<20} {exact:>2}/{len(CASES)}  {P:5.0%} {R:5.0%} {F1:5.0%}"
              f"{'  <- best' if name == rows[0][0] else ''}")
