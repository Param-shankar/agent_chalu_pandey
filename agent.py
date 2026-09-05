"""The dispute agent: normalized case in, merchant-facing recommendation out.

main.py hands us a dispute and the case tools.normalize_case already built from
the webhook. We give the model that case plus the Razorpay tools, let it call
them if it wants more than the webhook carried, then make one final call that
must answer in the VERDICT_SCHEMA shape.

Two rules the file is built around:
  - the model never invents evidence. It sees what the API returned and which
    evidence slots the dispute's own `evidence` object already fills, nothing else.
  - a model failure degrades to a MANUAL_REVIEW verdict with the error attached,
    because a webhook worker must always end up with something to write down.

Env:
  GEMINI_API_KEY  (or GOOGLE_API_KEY)  key for google-genai
  GEMINI_MODEL    default gemini-3.8-flash
"""
import json
import os
import re
import time
from pathlib import Path

from google import genai

import tools
from const import SYSTEM_PROMPT, VERDICT_SCHEMA

ROOT = Path(__file__).parent
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
MAX_TOOL_ROUNDS = 4                             # a dispute needs one or two fetches, not ten
RATE_LIMIT_RETRIES = 2                          # the free tier is 20 requests/minute


def _load_env(path=ROOT / ".env"):
    """main.py loads .env before importing us, but agent.py also runs standalone."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()

_client = None


def client():
    """Built on first use, so importing this module never needs a key."""
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY not set in environment")
        _client = genai.Client(api_key=key)
    return _client


# tools.py describes its tools in the Anthropic shape; genai wants
# {type, name, description, parameters}. Same schema, different key.
GENAI_TOOLS = [
    {
        "type": "function",
        "name": t["name"],
        "description": t["description"],
        "parameters": t["input_schema"],
    }
    for t in tools.TOOL_SCHEMAS
]


# ---------- prompt ----------
def build_case_prompt(dispute, normalized):
    """What the model sees. Facts only, labelled by where they came from."""
    return "\n".join([
        "A dispute has arrived. Investigate it and recommend what the merchant should do.",
        "",
        "## NORMALIZED CASE",
        "Built by our own code from the dispute: the reason code resolved against the "
        "card network rulebook, the evidence slots that rulebook requires, and which of "
        "those slots the dispute's own `evidence` object already fills. "
        "`coverage_percent` is weighted by slot importance, not a plain count.",
        json.dumps(normalized, indent=1, default=str) if normalized else "(not available)",
        "",
        "## RAW DISPUTE ENTITY",
        "Exactly what the Razorpay API returned. Nothing has been added to it.",
        json.dumps(dispute, indent=1, default=str) if dispute else "(not available)",
        "",
        "If anything above is missing or looks stale, call fetch_dispute_expanded_payment "
        "before deciding. A slot listed under `missing` means the dispute entity does not "
        "carry it - say what the merchant must supply and how to get it, do not conclude "
        "that it does not exist.",
    ])


FINAL_INSTRUCTION = (
    "Now give the merchant your answer as a single JSON object matching the required "
    "schema. Ground every claim in the case above: name only evidence that is actually "
    "present, and set submission_ready to true only if no required slot is missing. "
    "If the deadline is close or the evidence is thin, say so in risks. Write "
    "merchant_actions as instructions a merchant can act on today, not as categories."
)


# ---------- the loop ----------
def _create(gc, **kwargs):
    """interactions.create with a bounded retry on the per-minute rate limit."""
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            return gc.interactions.create(**kwargs)
        except Exception as e:                  # noqa: BLE001
            transient = "429" in str(e) or "too_many_requests" in str(e)
            if not transient or attempt == RATE_LIMIT_RETRIES:
                raise
            wait = 15.0
            m = re.search(r"retry in ([0-9.]+)s", str(e))
            if m:
                wait = min(float(m.group(1)) + 1, 60)
            time.sleep(wait)



def _dump(steps):
    """Interaction steps -> the param dicts the next request wants."""
    return [s.model_dump(by_alias=True, exclude_none=True) for s in steps or []]


def _run_tools(interaction, seen):
    """Execute every function_call this turn produced. Returns (result steps, log).

    `seen` memoises calls across rounds. A dispute fetch that failed once will
    fail the same way again, and without this the model spends every round
    retrying it instead of reasoning about what it already has.
    """
    results, calls = [], []
    for step in interaction.steps or []:
        if getattr(step, "type", None) != "function_call":
            continue
        args = step.arguments or {}
        key = (step.name, json.dumps(args, sort_keys=True, default=str))
        if key in seen:
            out = dict(seen[key])
            out["note"] = ("You already made this exact call this turn and got this "
                           "result. Do not call it again - decide with what you have.")
        else:
            out = tools.dispatch(step.name, args)
            seen[key] = out
        err = out.get("error") if isinstance(out, dict) else None
        calls.append({"name": step.name, "arguments": args, "error": err})
        results.append({
            "type": "function_result",
            "call_id": step.id,
            "name": step.name,           # the API rejects a result step without it
            "result": json.dumps(out, default=str),
            "is_error": bool(err),
        })
    return results, calls


def _parse_verdict(text):
    """The final answer as a dict. Tolerates a fenced block around the JSON."""
    if not text:
        raise ValueError("model returned no text")
    body = text.strip()
    if body.startswith("```"):
        body = body.split("```")[1]
        if body.lower().startswith("json"):
            body = body.split("\n", 1)[-1]
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in model output: {text[:200]}")
    return json.loads(body[start:end + 1])


def _final_call(gc, steps):
    """Ask for the verdict with the schema enforced. If the backend rejects the
    response_format we retry without it - the prompt asks for JSON either way."""
    kwargs = dict(model=MODEL, input=steps, system_instruction=SYSTEM_PROMPT)
    try:
        return _create(gc, response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": VERDICT_SCHEMA,
        }, **kwargs)
    except Exception as e:                      # noqa: BLE001
        if "429" in str(e) or "too_many_requests" in str(e):
            raise                               # a quota problem, not a schema problem
        return _create(gc, **kwargs)


def run_dispute(dispute, normalized=None):
    """Investigate one dispute and return the merchant-facing verdict.

    Returns {"verdict": {...}, "tool_calls": [...], "tool_rounds": n,
             "model": ..., "error": None}. On any model or transport failure the
    verdict degrades to MANUAL_REVIEW with the error recorded, so the caller
    always has a case file worth keeping.
    """
    if normalized is None and dispute:
        try:
            normalized = tools.normalize_case(dispute)
        except Exception:                       # noqa: BLE001 - prompt says so instead
            normalized = None

    steps = [{
        "type": "user_input",
        "content": [{"type": "text", "text": build_case_prompt(dispute, normalized)}],
    }]
    tool_calls, rounds, seen = [], 0, {}

    try:
        gc = client()

        # phase 1: let the model pull anything the webhook did not carry
        for rounds in range(1, MAX_TOOL_ROUNDS + 1):
            interaction = _create(
                gc,
                model=MODEL,
                input=steps,
                system_instruction=SYSTEM_PROMPT,
                tools=GENAI_TOOLS,
            )
            # interaction.steps is only this turn's steps, so accumulate
            steps = steps + _dump(interaction.steps)
            results, calls = _run_tools(interaction, seen)
            tool_calls.extend(calls)
            if not results:
                break
            steps.extend(results)

        # phase 2: one call whose only job is the structured verdict
        steps.append({
            "type": "user_input",
            "content": [{"type": "text", "text": FINAL_INSTRUCTION}],
        })
        verdict = _parse_verdict(_final_call(gc, steps).output_text)

    except Exception as e:                      # noqa: BLE001 - never lose the case
        detail = f"{type(e).__name__}: {e}"
        return {"verdict": _fallback_verdict(normalized, detail), "tool_calls": tool_calls,
                "tool_rounds": rounds, "model": MODEL, "error": detail}

    return {"verdict": verdict, "tool_calls": tool_calls, "tool_rounds": rounds,
            "model": MODEL, "error": None}


def _fallback_verdict(normalized, error):
    """A verdict shaped like the real one, so downstream code has no special case.
    It asks for a human, because the model never got to look at this dispute."""
    missing = ((normalized or {}).get("evidence_analysis") or {}).get("missing") or []
    return {
        "recommendation": "MANUAL_REVIEW",
        "confidence": "low",
        "headline": "Automated analysis failed - review this dispute by hand.",
        "dispute_summary": ((normalized or {}).get("dispute") or {}).get("reason_description") or "",
        "reasoning": f"The agent could not complete its analysis: {error}",
        "evidence_found": [],
        "missing_evidence": [{"type": slot,
                              "why_it_matters": "Required by the reason code for this dispute.",
                              "how_to_obtain": "Merchant must supply it."} for slot in missing],
        "risks": ["No model analysis was produced for this dispute."],
        "merchant_actions": [{"action": "Review this dispute manually before the deadline.",
                              "priority": "now",
                              "why": "The automated agent did not return a recommendation."}],
        "proposed_response": "",
        "submission_ready": False,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python agent.py disp_XXXXXXXX")
        raise SystemExit(2)
    print(json.dumps(run_dispute(tools.fetch_dispute_expanded_payment(sys.argv[1])),
                     indent=1, default=str))


# =====================================================================
# Talking to the agent about a case
# =====================================================================
def chat_start(dispute, normalized=None, verdict=None):
    """Opening history for a conversation about one dispute.

    Same facts the investigation saw, plus the verdict it produced, so the
    merchant can argue with the recommendation rather than re-explain the case.
    """
    text = build_case_prompt(dispute, normalized)
    if verdict:
        text += ("\n\n## THE RECOMMENDATION YOU ALREADY GAVE\n"
                 + json.dumps(verdict, indent=1, default=str)
                 + "\n\nThe merchant will now ask you about this dispute. Answer from the "
                   "case above. If they tell you about evidence that is not in the case, "
                   "treat it as a claim to verify, not as fact, and say what would confirm "
                   "it. Keep answers short and practical - a few sentences unless asked "
                   "for more. Plain text, not JSON.")
    return [{"type": "user_input", "content": [{"type": "text", "text": text}]},
            {"type": "user_input", "content": [{"type": "text",
             "text": "Ready. I will answer the merchant's questions about this dispute."}]}]


def chat_turn(history, message):
    """One turn. Returns (reply_text, new_history)."""
    steps = history + [{"type": "user_input",
                        "content": [{"type": "text", "text": message}]}]
    interaction = _create(client(), model=MODEL, input=steps,
                          system_instruction=SYSTEM_PROMPT, tools=GENAI_TOOLS)
    steps = steps + _dump(interaction.steps)

    # let it use the tools mid-conversation, same rules as the investigation
    seen = {}
    for _ in range(MAX_TOOL_ROUNDS):
        results, _calls = _run_tools(interaction, seen)
        if not results:
            break
        steps.extend(results)
        interaction = _create(client(), model=MODEL, input=steps,
                              system_instruction=SYSTEM_PROMPT, tools=GENAI_TOOLS)
        steps = steps + _dump(interaction.steps)

    return (interaction.output_text or "").strip(), steps
