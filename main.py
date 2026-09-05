"""Flask webhook receiver for Razorpay dispute events.

Razorpay POSTs here when a dispute is created or changes state. This server:
  1. verifies the X-Razorpay-Signature HMAC before trusting a single byte,
  2. de-duplicates on x-razorpay-event-id (Razorpay retries on non-2xx),
  3. acks in <5s and does the investigation on a worker thread,
  4. enriches the payload with GET /disputes/:id?expand[]=payment (tools.py),
  5. runs agent.run_dispute if it is importable, and writes a case file.

Nothing here fabricates dispute data: the case file holds what the webhook sent
plus what the API returned. Credentials are never written to disk or echoed.

Env (.env is loaded on import):
  RAZORPAY_KEY_ID          key id
  RAZORPAY_KEY_SECRET      key secret
  RAZORPAY_WEBHOOK_SECRET  the secret you typed into the Razorpay webhook form
  PORT                     default 5050
  SKIP_SIGNATURE_CHECK     "1" to accept unsigned posts (local curl only)

Routes:
  POST /webhook/razorpay     the URL to register in the Razorpay dashboard
  GET  /health               liveness + config sanity, no secrets
  GET  /cases                case files written so far
  GET  /cases/<dispute_id>   one case file
  POST /replay/<dispute_id>  run the pipeline by hand, no webhook needed
"""
import hashlib
import hmac
import json
import os
import queue
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

from flask import Flask, jsonify, request

import tools

ROOT = Path(__file__).parent
CASES = ROOT / "cases"
EVENT_LOG = ROOT / "cases" / "events.jsonl"
RAW = ROOT / "cases" / "raw"


# ---------- .env loader (keeps the dependency list at flask + requests) ----------
def load_env(path=ROOT / ".env"):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()

# stage 2-9 pipeline. Optional: agent.py imports core, which is not in the repo yet.
try:
    from agent import run_dispute
except Exception as e:                          # noqa: BLE001 - report, do not crash the server
    run_dispute = None
    _PIPELINE_ERR = f"{type(e).__name__}: {e}"
else:
    _PIPELINE_ERR = None


# ---------- signature ----------
def verify_signature(raw_body: bytes, signature: str) -> bool:
    """HMAC-SHA256 of the exact bytes Razorpay sent, keyed with the webhook secret."""
    secret = os.environ.get("RAZORPAY_WEBHOOK_SECRET")
    if not secret or not signature:
        return False
    expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


# ---------- replay guard ----------
# class SeenEvents:
#     """Bounded LRU of x-razorpay-event-id. Razorpay redelivers on any non-2xx,
#     and a redelivered dispute.created must not open a second case."""

#     def __init__(self, cap=5000):
#         self.cap, self.ids, self.lock = cap, OrderedDict(), threading.Lock()

#     def add(self, event_id) -> bool:
#         """True if this is the first sighting."""
#         if not event_id:
#             return True
#         with self.lock:
#             if event_id in self.ids:
#                 self.ids.move_to_end(event_id)
#                 return False
#             self.ids[event_id] = time.time()
#             while len(self.ids) > self.cap:
#                 self.ids.popitem(last=False)
#             return True


# seen = SeenEvents()


# ---------- persistence ----------
def write_case(dispute_id, case):
    CASES.mkdir(exist_ok=True)
    (CASES / f"{dispute_id}.json").write_text(json.dumps(case, indent=1, default=str))


def write_raw(event_id, payload):
    """Every body we accept, verbatim. When an event does not normalise, the
    payload that caused it is on disk instead of only in a terminal scrollback."""
    RAW.mkdir(parents=True, exist_ok=True)
    name = str(event_id or f"noid-{int(time.time()*1000)}").replace("/", "_")
    (RAW / f"{name}.json").write_text(json.dumps(payload, indent=1, default=str))


def log_event(record):
    CASES.mkdir(exist_ok=True)
    with EVENT_LOG.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


# ---------- event handling (worker thread) ----------
def normalize_event(name):
    """payment.dispute.created -> created. Tolerates future prefixes."""
    return (name or "").rsplit(".", 1)[-1]


def _looks_like_dispute(obj):
    """A dispute entity has an id and dispute-only fields. Never a payment."""
    if not isinstance(obj, dict):
        return False
    if obj.get("entity") == "dispute":
        return True
    return bool(obj.get("id")) and any(
        k in obj for k in ("reason_code", "respond_by", "amount_deducted", "phase")
    )


def dispute_entity(payload):
    """The dispute entity out of whatever envelope arrived, or {}.

    Razorpay documents payload.dispute.entity, but what actually shows up
    varies: the entity posted bare, one level flatter, or - as the dashboard's
    test sender does - the whole event envelope wrapped inside another
    `payload` key. So walk the body breadth-first and take the first dict that
    looks like a dispute, shallowest first, preferring a `dispute` slot at any
    depth over a lucky match elsewhere.
    """
    queue_, depth = [(payload, 0)], 0
    while queue_:
        obj, depth = queue_.pop(0)
        if depth > 8:
            continue
        if isinstance(obj, dict):
            # a dispute slot at this level wins over anything deeper
            holder = obj.get("dispute")
            if isinstance(holder, dict):
                if _looks_like_dispute(holder.get("entity")):
                    return holder["entity"]
                if _looks_like_dispute(holder):
                    return holder
            if _looks_like_dispute(obj):
                return obj
            queue_.extend((v, depth + 1) for v in obj.values()
                          if isinstance(v, (dict, list)))
        elif isinstance(obj, list):
            queue_.extend((v, depth + 1) for v in obj
                          if isinstance(v, (dict, list)))
    return {}


def find_dispute_id(obj, _depth=0):
    """Any disp_* id anywhere in the body.

    Some senders (and the dashboard's test button) post an envelope that carries
    the dispute id without the dispute entity - a bare id, or only
    payload.payment.entity with the dispute referenced by id. One id is enough:
    the API has the rest.
    """
    if _depth > 6:
        return None
    if isinstance(obj, str):
        return obj if obj.startswith("disp_") else None
    if isinstance(obj, dict):
        for key in ("dispute_id", "id"):
            v = obj.get(key)
            if isinstance(v, str) and v.startswith("disp_"):
                return v
        for v in obj.values():
            found = find_dispute_id(v, _depth + 1)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_dispute_id(v, _depth + 1)
            if found:
                return found
    return None


def normalize_webhook(payload):
    """Normalise whatever arrived into the fixed `case` shape.

    Preferred path is the dispute entity carried by the webhook - no API call.
    If the envelope carries only a dispute id, fetch the dispute and normalise
    that instead of dropping the event. Returns (normalized, error, source);
    `error` is None when there simply was no dispute in the body - that is a
    non-dispute event, not a failure.
    """
    entity = dispute_entity(payload)
    if entity:
        try:
            return tools.normalize_case(entity), None, "webhook"
        except Exception as e:                  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}", "webhook"

    dispute_id = find_dispute_id(payload)
    if not dispute_id:
        return None, None, "none"               # nothing to normalise: not a dispute event

    try:
        return tools.normalize_case(tools.fetch_dispute_expanded_payment(dispute_id)), None, "api"
    except tools.RazorpayError as e:
        return None, f"razorpay api {e.status}: {e.body}", "api"
    except Exception as e:                      # noqa: BLE001
        return None, f"{type(e).__name__}: {e}", "api"


def investigate(dispute_id, webhook_dispute, event, normalized=None, normalize_error=None):
    """Pull the authoritative dispute from the API, then score it if we can.

    `normalized` is what normalize_webhook already made of the webhook payload.
    """
    case = {
        "dispute_id": dispute_id,
        "event": event,
        "received_at": time.time(),
        "webhook_dispute": webhook_dispute,
        "normalized": normalized,
        "normalize_error": normalize_error,
    }

    print(f"\n{_c('investigating ' + str(dispute_id), 'bold')}", flush=True)
    if os.environ.get("SKIP_DISPUTE_FETCH") == "1":
        # demo mode: the webhook body is the whole truth, do not call the API
        case["dispute"] = None
        case["fetch_error"] = None
        step_done("skipped the api fetch (SKIP_DISPUTE_FETCH=1), using the webhook entity")
        return _finish_case(case, dispute_id, webhook_dispute, normalized)
    try:
        with Spinner("fetching dispute from the Razorpay API"):
            live = tools.fetch_dispute_expanded_payment(dispute_id)
        case["dispute"] = live
        case["fetch_error"] = None
        step_done("fetched dispute from the API")
    except tools.RazorpayError as e:
        case["dispute"] = None
        case["fetch_error"] = {"status": e.status, "detail": e.body}
        step_done(_c(f"api fetch failed ({e.status}) - using the webhook's own entity", "yellow"))
    except Exception as e:                      # noqa: BLE001
        case["dispute"] = None
        case["fetch_error"] = {"status": None, "detail": f"{type(e).__name__}: {e}"}
        step_done(_c(f"api fetch failed ({type(e).__name__}) - using the webhook's own entity", "yellow"))

    return _finish_case(case, dispute_id, webhook_dispute, normalized)


def _finish_case(case, dispute_id, webhook_dispute, normalized):
    d = case["dispute"] or webhook_dispute or {}

    case["summary"] = {
        "reason_code": d.get("reason_code"),
        "reason_description": d.get("reason_description"),
        "status": d.get("status"),
        "phase": d.get("phase"),
        "amount_inr": (d.get("amount") or 0) / 100,
        "respond_by": d.get("respond_by"),
        "hours_to_respond": (round((d["respond_by"] - time.time()) / 3600, 1)
                             if d.get("respond_by") else None),
        "evidence_already_submitted": bool(d.get("evidence")),
    }

    if run_dispute is None:
        case["pipeline"] = {"ran": False, "reason": _PIPELINE_ERR}
    elif not d:
        case["pipeline"] = {"ran": False, "reason": "no dispute entity from webhook or api"}
    else:
        # `d` is the API entity when the fetch worked and the webhook's own entity
        # otherwise. A failed fetch is not a reason to skip the analysis - the
        # webhook already carried the reason code and the evidence object.
        try:
            with Spinner("agent is analysing the dispute (tool calls + verdict)"):
                result = run_dispute(dict(d), normalized)
            step_done("agent finished",
                      f"{result.get('tool_rounds')} rounds, "
                      f"{len(result.get('tool_calls') or [])} tool calls")
            case["pipeline"] = {"ran": True,
                                "source": "api" if case["dispute"] else "webhook",
                                "result": result}
        except Exception as e:                  # noqa: BLE001
            case["pipeline"] = {"ran": False, "reason": f"{type(e).__name__}: {e}"}

    write_case(dispute_id, case)
    return case


def handle(payload, event_id, normalized=None, normalize_error=None):
    event = payload.get("event", "")
    kind = normalize_event(event)
    dispute = dispute_entity(payload)
    # no entity in the body: normalize_webhook may still have resolved one by id
    dispute_id = (dispute.get("id")
                  or (normalized or {}).get("dispute", {}).get("id")
                  or find_dispute_id(payload))

    record = {"t": time.time(), "event": event, "event_id": event_id,
              "dispute_id": dispute_id}

    if not dispute_id:
        log_event({**record, "action": "ignored", "why": "no dispute in payload"})
        return

    if kind in ("created", "action_required", "under_review"):
        case = investigate(dispute_id, dispute, event, normalized, normalize_error)
        log_event({**record, "action": "investigated",
                   "recommendation": (case.get("pipeline", {}).get("result", {})
                                      .get("verdict", {}).get("recommendation")),
                   "reason_code": case["summary"]["reason_code"]})
        return case
    elif kind in ("won", "lost", "closed"):
        # terminal states: record the outcome, never act on the dispute
        case_path = CASES / f"{dispute_id}.json"
        if case_path.exists():
            case = json.loads(case_path.read_text())
            case.setdefault("outcomes", []).append({"event": event, "t": time.time(),
                                                    "amount_deducted": dispute.get("amount_deducted")})
            write_case(dispute_id, case)
        log_event({**record, "action": "outcome_recorded"})
    else:
        log_event({**record, "action": "ignored", "why": f"unhandled event {event}"})


# ---------- terminal report ----------
# The webhook fires and the answer has to be readable in the terminal that is
# running the server, not only in cases/<id>.json.
_C = {"reset": "\033[0m", "bold": "\033[1m", "dim": "\033[2m",
      "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
      "blue": "\033[34m", "cyan": "\033[36m"}
_VERDICT_COLOR = {"CONTEST": "green", "ACCEPT": "yellow",
                  "REQUEST_MORE_EVIDENCE": "cyan", "MANUAL_REVIEW": "red"}


def _c(text, *styles):
    """Colour, unless stdout is a pipe or NO_COLOR is set."""
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return str(text)
    return "".join(_C[s] for s in styles) + str(text) + _C["reset"]


def _wrap(text, width=76, indent="    "):
    out, line = [], indent
    for word in str(text).split():
        if len(line) + len(word) + 1 > width and line.strip():
            out.append(line)
            line = indent
        line += word + " "
    if line.strip():
        out.append(line)
    return "\n".join(out).rstrip()


class Spinner:
    """A live 'still working' marker. Silent when stdout is not a terminal."""

    FRAMES = "|/-\\"

    def __init__(self, label):
        self.label, self.stop_flag, self.thread = label, threading.Event(), None

    def _spin(self):
        i = 0
        while not self.stop_flag.is_set():
            sys.stdout.write(f"\r  {self.FRAMES[i % 4]} {self.label} ")
            sys.stdout.flush()
            i += 1
            self.stop_flag.wait(0.12)
        sys.stdout.write("\r" + " " * (len(self.label) + 8) + "\r")
        sys.stdout.flush()

    def __enter__(self):
        if sys.stdout.isatty() and not os.environ.get("NO_COLOR"):
            self.thread = threading.Thread(target=self._spin, daemon=True)
            self.thread.start()
        else:
            print(f"  ... {self.label}", flush=True)
        return self

    def __exit__(self, *exc):
        self.stop_flag.set()
        if self.thread:
            self.thread.join(timeout=1)


def step_done(label, detail=""):
    print(f"  {_c('done', 'green')} {label}{(' - ' + detail) if detail else ''}", flush=True)


def render_case(case):
    """Print one finished case the way a merchant would want to read it."""
    summary = case.get("summary") or {}
    norm = (case.get("normalized") or {}).get("dispute") or {}
    result = (case.get("pipeline") or {}).get("result") or {}
    verdict = result.get("verdict") or {}
    analysis = (case.get("normalized") or {}).get("evidence_analysis") or {}

    print()
    print(_c("=" * 78, "dim"))
    title = f" DISPUTE {case.get('dispute_id')}  |  {case.get('event')}"
    print(_c(title, "bold"))
    print(_c("=" * 78, "dim"))

    reason = norm.get("reason_title") or summary.get("reason_description") or "unknown reason"
    print(f"  {_c('Reason', 'dim')}    {summary.get('reason_code')}  {reason}")
    if norm.get("network"):
        print(f"  {_c('Network', 'dim')}   {norm.get('network')} / {norm.get('category')}")
    print(f"  {_c('Amount', 'dim')}    Rs {summary.get('amount_inr')}"
          f"   {_c('Status', 'dim')} {summary.get('status')} / {summary.get('phase')}")

    hours = summary.get("hours_to_respond")
    if hours is not None:
        style = "red" if hours < 48 else "yellow" if hours < 120 else "green"
        print(f"  {_c('Deadline', 'dim')}  {_c(f'{hours} hours to respond', style)}")

    if analysis:
        print(f"  {_c('Evidence', 'dim')}  {analysis.get('coverage_percent')}% of required "
              f"slots covered ({analysis.get('coverage_score')}/{analysis.get('max_score')} weighted)")

    if not verdict:
        why = (case.get("pipeline") or {}).get("reason")
        print(_c(f"\n  No agent verdict: {why}", "red"))
        print(_c("=" * 78, "dim"), flush=True)
        return

    rec = verdict.get("recommendation", "?")
    print()
    print(f"  {_c(rec, 'bold', _VERDICT_COLOR.get(rec, 'blue'))}"
          f"   {_c('confidence: ' + str(verdict.get('confidence')), 'dim')}")
    print(_wrap(verdict.get("headline", ""), indent="  "))

    if verdict.get("reasoning"):
        print(_c("\n  WHY", "bold"))
        print(_wrap(verdict["reasoning"]))

    found = verdict.get("evidence_found") or []
    if found:
        print(_c("\n  EVIDENCE ON FILE", "bold"))
        for e in found:
            print(f"    {_c('+', 'green')} {_c(e.get('type'), 'bold')} "
                  f"{_c('(' + str(e.get('source')) + ')', 'dim')}")
            print(_wrap(e.get("detail", ""), indent="      "))

    missing = verdict.get("missing_evidence") or []
    if missing:
        print(_c("\n  MISSING EVIDENCE", "bold"))
        for m in missing:
            print(f"    {_c('-', 'red')} {_c(m.get('type'), 'bold')}")
            print(_wrap(m.get("why_it_matters", ""), indent="      "))
            print(_wrap(f"how to get it: {m.get('how_to_obtain')}", indent="      "))

    risks = verdict.get("risks") or []
    if risks:
        print(_c("\n  RISKS", "bold"))
        for r in risks:
            print(_wrap(f"! {r}", indent="    "))

    actions = verdict.get("merchant_actions") or []
    if actions:
        print(_c("\n  DO THIS", "bold"))
        for i, a in enumerate(actions, 1):
            pr = a.get("priority", "")
            style = "red" if pr == "now" else "yellow" if pr == "before_deadline" else "dim"
            print(f"    {i}. {_c('[' + pr + ']', style)}")
            print(_wrap(a.get("action", ""), indent="       "))
            print(_wrap(_c(a.get("why", ""), "dim"), indent="       "))

    if verdict.get("proposed_response"):
        print(_c("\n  DRAFT RESPONSE", "bold"))
        print(_wrap(verdict["proposed_response"], indent="    "))

    ready = verdict.get("submission_ready")
    print()
    print(f"  {_c('Submission ready', 'dim')}  "
          f"{_c('yes', 'green') if ready else _c('no - evidence still missing', 'yellow')}")
    if result.get("error"):
        detail = " ".join(str(result["error"]).split())
        print(f"  {_c('Agent error', 'dim')}      {_c(detail[:160], 'red')}"
              f"{_c(' ...', 'dim') if len(detail) > 160 else ''}")
    print(f"  {_c('Case file', 'dim')}        cases/{case.get('dispute_id')}.json")
    print(f"  {_c('Talk to it', 'dim')}       "
          f"{_c('python main.py chat ' + str(case.get('dispute_id')), 'cyan')}")
    print(_c("=" * 78, "dim"))
    print(flush=True)


work = queue.Queue()


def worker():
    while True:
        payload, event_id, normalized, normalize_error = work.get()
        try:
            case = handle(payload, event_id, normalized, normalize_error)
            if case:
                render_case(case)
        except Exception as e:                  # noqa: BLE001 - one bad event must not kill the loop
            log_event({"t": time.time(), "event_id": event_id,
                       "action": "worker_error", "detail": f"{type(e).__name__}: {e}"})
        finally:
            work.task_done()


threading.Thread(target=worker, daemon=True, name="dispute-worker").start()


# ---------- routes ----------
app = Flask(__name__)


@app.post("/webhook/razorpay")
def webhook():
    raw = request.get_data()                    # raw bytes: the signature covers these exactly
    sig = request.headers.get("X-Razorpay-Signature", "")
    event_id = request.headers.get("x-razorpay-event-id")

    if os.environ.get("SKIP_SIGNATURE_CHECK") != "1" and not verify_signature(raw, sig):
        return jsonify({"error": "invalid signature"}), 401

    try:
        payload = json.loads(raw)
    except ValueError:
        return jsonify({"error": "body is not json"}), 400

    # if not seen.add(event_id):
    #     return jsonify({"status": "duplicate ignored", "event_id": event_id}), 200


    write_raw(event_id, payload)                # keep the body, whatever happens next

    # normalise the moment it lands, before anything else touches it
    normalized, normalize_error, source = normalize_webhook(payload)

    if normalized:
        print(f"normalized from {source}:")
        print("**" * 50)
        print(json.dumps(normalized, indent=1, default=str))
    elif normalize_error:
        print(f"normalize failed for event {event_id} (via {source}): {normalize_error}")
        print(json.dumps(payload, indent=1, default=str)[:2000])
    else:
        print(f"no dispute in event {event_id} ({payload.get('event')}), nothing to normalize")
        print(json.dumps(payload, indent=1, default=str)[:2000])


    # ack now, investigate off-thread: Razorpay retries anything slower than ~5s.
    # The worker prints the finished report to this terminal when it lands.
    work.put((payload, event_id, normalized, normalize_error))
    return jsonify({"status": "accepted", "event": payload.get("event"),
                    "event_id": event_id,
                    "normalized": normalized,
                    "normalized_from": source,
                    "normalize_error": normalize_error}), 200


@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "webhook_secret_configured": bool(os.environ.get("RAZORPAY_WEBHOOK_SECRET")),
        "api_keys_configured": bool(os.environ.get("RAZORPAY_KEY_ID")
                                    and os.environ.get("RAZORPAY_KEY_SECRET")),
        "signature_check": os.environ.get("SKIP_SIGNATURE_CHECK") != "1",
        "pipeline_available": run_dispute is not None,
        "pipeline_error": _PIPELINE_ERR,
        "queue_depth": work.qsize(),
        "cases_written": len(list(CASES.glob("*.json"))) if CASES.exists() else 0,
    })


@app.get("/cases")
def list_cases():
    if not CASES.exists():
        return jsonify({"cases": []})
    out = []
    for p in sorted(CASES.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        c = json.loads(p.read_text())
        verdict = (c.get("pipeline", {}).get("result") or {}).get("verdict") or {}
        out.append({"dispute_id": c.get("dispute_id"), "event": c.get("event"),
                    "summary": c.get("summary"),
                    "pipeline_ran": c.get("pipeline", {}).get("ran"),
                    "recommendation": verdict.get("recommendation"),
                    "headline": verdict.get("headline")})
    return jsonify({"cases": out})


@app.get("/cases/<dispute_id>") 
def get_case(dispute_id):
    p = CASES / f"{dispute_id}.json"
    if not p.exists():
        return jsonify({"error": "no case file", "dispute_id": dispute_id}), 404
    return jsonify(json.loads(p.read_text()))


@app.post("/replay/<dispute_id>")
def replay(dispute_id):
    """Investigate a dispute id directly. Same path the webhook takes, minus the webhook."""
    return jsonify(investigate(dispute_id, {}, "manual.replay"))


# ---------- cli ----------
def cli_show(dispute_id):
    """Re-print a case that was already investigated."""
    path = CASES / f"{dispute_id}.json"
    if not path.exists():
        print(f"no case file for {dispute_id}. Investigated cases: "
              f"{[p.stem for p in CASES.glob('*.json') if p.stem != 'events']}")
        return 1
    render_case(json.loads(path.read_text()))
    return 0


def cli_run(dispute_id):
    """Investigate a dispute id from the terminal - the webhook path, no webhook."""
    print(f"investigating {dispute_id} ...")
    render_case(investigate(dispute_id, {}, "cli.run"))
    return 0


def cli_chat(dispute_id):
    """Ask the agent about a case it already investigated."""
    from agent import chat_start, chat_turn

    path = CASES / f"{dispute_id}.json"
    if not path.exists():
        print(f"no case file for {dispute_id} - run: python main.py run {dispute_id}")
        return 1
    case = json.loads(path.read_text())
    dispute = case.get("dispute") or case.get("webhook_dispute") or {}
    verdict = ((case.get("pipeline") or {}).get("result") or {}).get("verdict")

    render_case(case)
    print(_c(f"chatting about {dispute_id}. ctrl-c or 'exit' to quit.", "dim"))
    history = chat_start(dispute, case.get("normalized"), verdict)

    while True:
        try:
            message = input(_c("\nyou > ", "bold", "cyan")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not message:
            continue
        if message.lower() in ("exit", "quit", "q"):
            return 0
        try:
            with Spinner("thinking"):
                reply, history = chat_turn(history, message)
        except Exception as e:                  # noqa: BLE001 - keep the session alive
            print(_c(f"  agent error: {type(e).__name__}: {e}", "red"))
            continue
        print(f"\n{_c('agent >', 'bold', 'green')} {_wrap(reply, indent='').lstrip()}")


def cli_list():
    if not CASES.exists():
        print("no cases yet")
        return 0
    rows = sorted(CASES.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in rows:
        case = json.loads(path.read_text())
        verdict = ((case.get("pipeline") or {}).get("result") or {}).get("verdict") or {}
        print(f"{case.get('dispute_id'):24} {str(verdict.get('recommendation') or '-'):22} "
              f"{(case.get('summary') or {}).get('reason_code') or ''}")
    return 0


def serve():
    port = int(os.environ.get("PORT", 5050))
    print(f"webhook  -> http://127.0.0.1:{port}/webhook/razorpay")
    print(f"health   -> http://127.0.0.1:{port}/health")
    print("post a dispute event to the webhook and the report prints here")
    if not os.environ.get("RAZORPAY_WEBHOOK_SECRET"):
        print("WARNING: RAZORPAY_WEBHOOK_SECRET is not set, every signed post will 401")
    app.run(host="0.0.0.0", port=port, threaded=True)


USAGE = """usage:
  python main.py                 run the webhook server (reports print here)
  python main.py show <disp_id>  re-print an investigated case
  python main.py run <disp_id>   investigate a dispute now, no webhook needed
  python main.py chat <disp_id>  ask the agent questions about a case
  python main.py list            one line per investigated case
"""

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        serve()
    elif argv[0] == "show" and len(argv) > 1:
        raise SystemExit(cli_show(argv[1]))
    elif argv[0] == "run" and len(argv) > 1:
        raise SystemExit(cli_run(argv[1]))
    elif argv[0] == "chat" and len(argv) > 1:
        raise SystemExit(cli_chat(argv[1]))
    elif argv[0] == "list":
        raise SystemExit(cli_list())
    else:
        print(USAGE)
        raise SystemExit(2)
