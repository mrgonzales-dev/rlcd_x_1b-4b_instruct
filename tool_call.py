import json
import httpx

# ── SERVER ────────────────────────────────────────────────────────────────
# Requires load_laya.py running:  .venv/bin/python load_laya.py
SERVER = "http://127.0.0.1:8000"
MODEL = "english"

# ── TOOLS the llm could call ───────────────────────────────────────────────
TOOLS = {
    "none": "answer directly, no tool needed",
    "get_weather": "current weather or forecast for a location",
    "web_search": "look up facts, news, or information on the internet",
    "calculator": "evaluate a math expression",
    "send_email": "compose and send an email",
}

# ── CASES ─────────────────────────────────────────────────────────────────
# state: the user message an llm is about to answer. expect: the correct tool.
CASES = [
    {"state": "Hey, what's the weather like in Tokyo right now?", "expect": "get_weather"},
    {"state": "Will it rain in London this weekend?", "expect": "get_weather"},
    {"state": "Do I need a jacket in Berlin tomorrow morning?", "expect": "get_weather"},
    {"state": "What are today's top headlines?", "expect": "web_search"},
    {"state": "Search for the current price of Bitcoin.", "expect": "web_search"},
    {"state": "Who won yesterday's Champions League match?", "expect": "web_search"},
    {"state": "What is 128 * 46 + 17?", "expect": "calculator"},
    {"state": "Compute a 15% tip on an $84.50 bill.", "expect": "calculator"},
    {"state": "Send an email to john@acme.com saying the meeting moved to 3pm.", "expect": "send_email"},
    {"state": "Email my landlord that the kitchen sink is leaking.", "expect": "send_email"},
    {"state": "What is the capital of France?", "expect": "none"},
    {"state": "Explain how photosynthesis works.", "expect": "none"},
    {"state": "Write a haiku about autumn.", "expect": "none"},
    {"state": "Hi! How are you today?", "expect": "none"},
]

# ── QUESTIONS ─────────────────────────────────────────────────────────────
QUESTIONS = {
    # noul: should a tool be called at all?
    "needs_tool": {
        "type": "noul",
        "instructions": "Should the assistant call a tool (weather, search, calculator, email) to answer this request?",
    },
    # choice: which tool?
    "tool": {
        "type": "choice",
        "instructions": "Which tool should be called to answer this request?",
        "criteria": TOOLS,
    },
    # score: how necessary is a tool, low -> high
    "tool_necessity": {
        "type": "score",
        "instructions": "How necessary is a tool call to answer this correctly?",
        "criteria": ["can answer from knowledge", "tool would help", "tool is required"],
    },
}

# ── DECISION RULES ────────────────────────────────────────────────────────
# Each rule maps laya's answers -> chosen tool. Compared against "expect".
RULES = {
    "choice_only":   lambda a: a["tool"]["choice"],
    "noul+choice":   lambda a: a["tool"]["choice"] if a["needs_tool"]["noul"] > 0.5 else "none",
    "score+choice":  lambda a: a["tool"]["choice"] if a["tool_necessity"]["score"] >= 1.0 else "none",
    "noul|choice":   lambda a: "none" if a["tool"]["choice"] == "none" and a["needs_tool"]["noul"] <= 0.5
                                 else a["tool"]["choice"],
}

# ── RUN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    hits = {name: 0 for name in RULES}
    totals, tokens, ms = 0, 0, 0

    for i, case in enumerate(CASES):
        response = httpx.post(f"{SERVER}/evaluate",
                              json={"state": case["state"], "questions": QUESTIONS, "model": MODEL},
                              timeout=120)
        result = response.json()
        if response.status_code != 200:
            raise SystemExit(f"{SERVER} returned {response.status_code}: {result.get('error')}")
        answers, expect = result["answers"], case["expect"]
        totals += 1
        tokens += result["usage"]["input_tokens"]
        ms += result["elapsed_ms"]

        print(f"\n{'=' * 70}\n[{i}] {case['state'][:66]}\n  expect: {expect}"
              f"  | needs_tool {answers['needs_tool']['noul']:.2f}"
              f"  | tool {answers['tool']['choice']} ({answers['tool']['probabilities'][answers['tool']['choice']]:.2f})"
              f"  | necessity {answers['tool_necessity']['score']:.2f}")
        for name, rule in RULES.items():
            pick = rule(answers)
            ok = pick == expect
            hits[name] += ok
            print(f"  {name:<13} -> {pick:<12} {'ok' if ok else 'MISS'}")

    # ── METRICS: which rule decides correctly most of the time ─────────────
    print(f"\n{'=' * 70}\naccuracy over {totals} cases:")
    best = max(RULES, key=lambda n: hits[n])
    for name in RULES:
        bar = "#" * hits[name] + "-" * (totals - hits[name])
        print(f"  {name:<13} {hits[name]:>2}/{totals}  {hits[name] / totals:5.0%}  {bar}"
              f"{'  <- best' if name == best else ''}")
    print(f"\n[{tokens} input tokens total, {ms / totals:.0f} ms avg]")
