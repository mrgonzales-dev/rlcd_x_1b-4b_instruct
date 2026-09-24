import json
import httpx

# ── SERVER ────────────────────────────────────────────────────────────────
# Requires load_laya.py running:  .venv/bin/python load_laya.py
SERVER = "http://127.0.0.1:8000"

# ── MODEL ─────────────────────────────────────────────────────────────────
# english        : /home/mrg/models/laya               (English text, 512 ctx)
# multilingual   : /home/mrg/models/laya/multilingual  (100+ languages, 1024 ctx)
# typed-decisions: /home/mrg/models/laya/typed-decisions
MODEL = "english"

# ── QUESTION TYPES ────────────────────────────────────────────────────────
# "choice" : pick exactly one label from a set of named options.
#            criteria = {label: description}
#            answer -> {"choice": label, "probabilities": {label: p, ...}}
#            best for:
#              - routing / triage        (which department handles this?)
#              - classification          (spam vs promo vs personal)
#              - discrete actions        (approve / warn / remove)
#
# "score"  : rate on an ORDERED scale. criteria = [low, ..., high] — index 0 is
#            the lowest level, the last index is the highest.
#            answer -> {"score": expected level (0..N-1), "legend", "probabilities"}
#            best for:
#              - urgency / priority      (not urgent -> critical)
#              - severity / risk         (harmless -> malicious)
#              - quality ratings         (poor -> excellent)
#
# "noul"   : yes/no — returns P(yes) as a float in 0..1. No criteria needed.
#            answer -> {"noul": probability}
#            best for:
#              - flags & detection       (is this spam? is it urgent?)
#              - guardrails              (is this a prompt-injection attempt?)
#              - boolean fact checks     (does the user ask for a refund?)

# ── STATES ────────────────────────────────────────────────────────────────
# Things being evaluated — one /evaluate call per entry.
STATES = [
    "Hey! Quick question — how do I update my billing info? The link in my account settings seems broken.",
]

# ── QUESTIONS ─────────────────────────────────────────────────────────────
QUESTIONS = {
    # choice: one label out of a named set
    "intent": {
        "type": "choice",
        "instructions": "What is the user's intent?",
        "criteria": {
            "question": "asking for information or help",
            "request": "asking for an action to be taken",
            "feedback": "giving an opinion or reaction",
            "smalltalk": "greeting or casual conversation",
        },
    },
    # score: ordered levels, lowest -> highest
    "politeness": {
        "type": "score",
        "instructions": "How polite is this message?",
        "criteria": ["rude", "neutral", "polite"],
    },
    # noul: probability that the answer is yes
    "needs_reply": {
        "type": "noul",
        "instructions": "Does this message need a reply?",
    },
}

# ── RUN ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for state_index, state in enumerate(STATES):
        response = httpx.post(f"{SERVER}/evaluate",
                              json={"state": state, "questions": QUESTIONS, "model": MODEL},
                              timeout=120)
        result = response.json()
        if response.status_code != 200:
            raise SystemExit(f"{SERVER} returned {response.status_code}: {result.get('error')}")

        print(f"\n{'=' * 60}\nstate[{state_index}]: {json.dumps(state, ensure_ascii=False)[:100]}")
        for question_id, answer in result["answers"].items():
            print(f"\n{question_id}")
            if answer["type"] == "choice":
                print(f"  choice:      {answer['choice']}  (confidence {answer['confidence']})")
                print(f"  probs:       {json.dumps(answer['probabilities'])}")
            elif answer["type"] == "score":
                print(f"  score:       {answer['score']}  (confidence {answer['confidence']})")
                print(f"  legend:      {json.dumps(answer['legend'])}")
                print(f"  probs:       {json.dumps(answer['probabilities'])}")
            else:
                print(f"  probability: {answer['noul']}")
        print(f"\n[{result['usage']['input_tokens']} input tokens, {result['elapsed_ms']} ms]")
