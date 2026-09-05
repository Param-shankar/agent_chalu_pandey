"""Razorpay API tools exposed to the dispute agent.

Every tool returns plain JSON-serialisable dicts. Nothing here fabricates data:
if the API does not return a field, the field is absent. Credentials are read
from the environment and are never echoed back into a tool result (see
SYSTEM_PROMPT rule 8).

Env:
  RAZORPAY_KEY_ID      key id      (rzp_test_... / rzp_live_...)
  RAZORPAY_KEY_SECRET  key secret
  RAZORPAY_API_BASE    optional, defaults to https://api.razorpay.com/v1
"""
import os
import requests

API_BASE = os.environ.get("RAZORPAY_API_BASE", "https://api.razorpay.com/v1").rstrip("/")
TIMEOUT = 20


class RazorpayError(RuntimeError):
    """API call failed. Carries status + Razorpay error body, never the credentials."""

    def __init__(self, status, body):
        self.status, self.body = status, body
        super().__init__(f"razorpay api {status}: {body}")


def _auth():
    kid, secret = os.environ.get("RAZORPAY_KEY_ID"), os.environ.get("RAZORPAY_KEY_SECRET")
    if not kid or not secret:
        raise RuntimeError("RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET not set in environment")
    return (kid, secret)


def _get(path, params=None):
    r = requests.get(f"{API_BASE}{path}", auth=_auth(), params=params,
                     headers={"Content-Type": "application/json"}, timeout=TIMEOUT)
    try:
        body = r.json()
    except ValueError:
        body = {"error": {"description": r.text[:500]}}
    if not r.ok:
        raise RazorpayError(r.status_code, body.get("error", body))
    # some hosts wrap the entity in {status_code, success, data}
    return body.get("data", body) if isinstance(body, dict) and "data" in body else body


# ---------- GET /disputes/:id?expand[]=payment ----------
def fetch_dispute_expanded_payment(dispute_id, expand=("payment",)):
    """Fetch one dispute with the payment entity expanded inline.

    https://razorpay.com/docs/api/disputes/fetch-dispute-expanded-payment

    Args:
        dispute_id: e.g. "disp_K8bVLppJ8zp5Wp".
        expand: entities to inline. "payment" replaces the payment_id string
            with the full payment object; pass ("payment", "payment.card") to
            also inline the card used.

    Returns the dispute entity: id, entity, payment (expanded) / payment_id,
    amount and amount_deducted (in paise), currency, status, phase,
    reason_code, reason_description, respond_by, evidence, lifecycle,
    created_at.

    Raises RazorpayError on a non-2xx response (404 = unknown dispute id).
    """
    if not dispute_id:
        raise ValueError("dispute_id is required")
    # Razorpay takes repeated expand[] keys, not a comma joined list
    params = [("expand[]", e) for e in expand]
    return _get(f"/disputes/{dispute_id}", params=params)


# ---------- Anthropic tool-use schemas ----------
FETCH_DISPUTE_EXPANDED_PAYMENT = {
    "name": "fetch_dispute_expanded_payment",
    "description": (
        "Fetch a Razorpay dispute by id with the associated payment expanded inline. "
        "Use this as the first call in an investigation to get the authoritative "
        "reason_code, phase, status, respond_by deadline, already-submitted evidence, "
        "and the underlying payment (method, status, captured amount, email/contact). "
        "Amounts are in paise. Returns only what the API returns."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "dispute_id": {
                "type": "string",
                "description": "Razorpay dispute id, e.g. disp_K8bVLppJ8zp5Wp",
            },
            "expand": {
                "type": "array",
                "items": {"type": "string", "enum": ["payment", "payment.card", "payment.offers"]},
                "description": "Entities to inline. Defaults to [\"payment\"].",
            },
        },
        "required": ["dispute_id"],
    },
}

TOOL_SCHEMAS = [FETCH_DISPUTE_EXPANDED_PAYMENT]
TOOL_IMPLS = {"fetch_dispute_expanded_payment": fetch_dispute_expanded_payment}


def dispatch(name, tool_input):
    """Run a tool by name. Errors come back as data so the agent can reason about
    them instead of the loop dying mid-investigation."""
    fn = TOOL_IMPLS.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(**tool_input)
    except RazorpayError as e:
        return {"error": "razorpay_api_error", "status": e.status, "detail": e.body}
    except Exception as e:                      # never leak a traceback with creds in it
        return {"error": type(e).__name__, "detail": str(e)}


if __name__ == "__main__":
    import json, sys
    print(json.dumps(fetch_dispute_expanded_payment(sys.argv[1]), indent=1))


# =====================================================================
# Webhook -> LLM case normalisation
# =====================================================================
"""Turns whatever the webhook + GET /disputes/:id gave us into the fixed
`case` shape the system prompt consumes.

The mapping chain is:

    dispute.reason_code  --reasoncode.json-->  required documents (free text)
                         --DOCUMENT_RULES-->   Razorpay evidence slots
                         --dispute.evidence--> coverage score / missing list

Nothing is invented: a field the API did not return stays None, and an
unrecognised reason code produces an empty requirement list rather than a
guessed one.
"""

import json as _json
import math as _math
import re as _re
import time as _time
from pathlib import Path as _Path

REASON_CODES_PATH = _Path(__file__).parent / "reasoncode.json"

# Razorpay's contest-dispute evidence slots. `others` is the catch-all the API
# takes as a list of {type, custom_type, document_ids}.
EVIDENCE_SLOTS = (
    "shipping_proof",
    "proof_of_service",
    "customer_communication",
    "access_activity_log",
    "refund_confirmation",
    "cancellation_proof",
    "billing_proof",
    "refund_cancellation_policy",
    "term_and_conditions",
    "explanation_letter",
    "others",
)

# How much a slot moves the needle when contesting. Primary proof that the
# merchant did the thing outweighs the boilerplate policy documents.
SLOT_WEIGHT = {
    "shipping_proof": 3,
    "proof_of_service": 3,
    "access_activity_log": 3,
    "refund_confirmation": 3,
    "customer_communication": 2,
    "cancellation_proof": 2,
    "billing_proof": 1,
    "refund_cancellation_policy": 1,
    "term_and_conditions": 1,
    "explanation_letter": 1,
    "others": 1,
}

# reasoncode.json lists documents as free text ("Delivery confirmation with
# signature"). These rules fold those 300-odd phrases onto the slots above.
# ORDER MATTERS: first rule that matches a phrase wins, so the specific
# phrases sit above the generic keywords.
DOCUMENT_RULES = [
    ("refund_cancellation_policy", (
        "refund polic", "return polic", "cancellation polic", "refund/cancellation",
        "cancellation window", "cancellation deadline", "no-show policy",
        "policy compliance", "restocking",
    )),
    ("term_and_conditions", (
        "terms", "contract", "policy agreement", "rental agreement",
        "compliance with oct rules",
    )),
    ("customer_communication", (
        "customer communication", "customer interaction", "customer acknowledgement",
        "customer confirmation", "customer consent", "customer agreement",
        "customer signature", "customer verification", "customer withdrawn",
        "customer did not contact", "no complaint received", "no consent given",
        "no prior agreement", "no misleading claims", "display proof",
        "display screenshots", "shopping cart screenshot",
    )),
    ("refund_confirmation", (
        "refund", "credit issuance", "credit timestamp", "void/refund",
        "credit not", "loss mitigation",
    )),
    ("cancellation_proof", (
        "cancellation", "cancelled", "no cancellation", "return not received",
        "no return received", "no damage claim", "return received",
    )),
    ("shipping_proof", (
        "shipping", "delivery", "deliver", "tracking", "courier", "dispatch",
        "shipped", "packaging", "order already processed", "already shipped",
    )),
    ("proof_of_service", (
        "service", "usage", "quality", "product description", "product/image",
        "product source", "photos", "specifications", "damage documentation",
        "authenticity", "accurate description", "accurate marketing",
        "supplier verification", "pre-rental inspection", "insurance",
        "reservation confirmation", "theft documentation", "police report",
    )),
    ("access_activity_log", (
        "log", "ip address", "ip geolocation", "device", "fingerprint",
        "3d secure", "avs", "cvv", "cvc", "pin verification", "emv", "chip",
        "terminal", "authentication", "authorisation", "authorised", "auth ",
        "timestamp", "time stamp", "system", "batch", "transaction id",
        "unique identifier", "unique reference", "fraud screening",
        "fraud prevention", "risk assessment", "card imprint", "card-present",
        "cardholder verification", "card number verification",
        "valid card verification", "number validation", "signature on file",
        "security footage", "account verification", "account match",
        "account confirmation", "data integrity", "correct transaction",
        "corrected transaction", "correction documentation", "processing record",
        "processing confirmation", "processing timeline", "settlement timing",
        "transaction approval", "transaction reversal", "transaction type",
        "proof of single payment", "single charge", "single payment",
        "single transaction", "multiple auth", "resubmission", "fallback",
        "compliance", "legitimate transaction", "business legitimacy",
        "crb check", "purchase pattern", "approval documentation",
        "approval records", "contactless approval", "valid data",
        "valid processing", "system configuration",
    )),
    ("billing_proof", (
        "invoice", "receipt", "billing", "price", "pricing", "amount",
        "charge", "bank statement", "payment reconciliation", "order record",
        "order history", "purchase history", "sale documentation", "currency",
        "exchange rate", "installment", "payment schedule",
        "payment method selection", "no alternative payment", "original date",
    )),
]

# Visa publishes dotted codes (13.1, 10.4, 12.6.1). reasoncode.json stores the
# sub-code under its dispute category, so the leading number is the category.
_VISA_CATEGORY_BY_PREFIX = {
    "10": "Fraud",
    "11": "Authorisation Error",
    "12": "Processing Error",
    "13": "Customer Dispute",
}

# What the webhook / card entity calls a network vs. the reasoncode.json key.
_NETWORK_ALIASES = {
    "visa": "Visa",
    "mastercard": "Mastercard",
    "master card": "Mastercard",
    "mc": "Mastercard",
    "rupay": "Rupay",
    "amex": "American Express",
    "american express": "American Express",
    "upi": "UPI",
    "razorpay": "Razorpay",
    "rzp": "Razorpay",
}

_REASON_CODES_CACHE = None


def load_reason_codes(path=REASON_CODES_PATH):
    """reasoncode.json, read once and memoised."""
    global _REASON_CODES_CACHE
    if _REASON_CODES_CACHE is None:
        _REASON_CODES_CACHE = _json.loads(_Path(path).read_text())
    return _REASON_CODES_CACHE


def detect_network(dispute, payment=None):
    """Which scheme's rulebook applies. card.network wins, then the payment
    method, then the reason-code prefix. Returns None if nothing says."""
    payment = payment or {}
    card = payment.get("card") or {}
    raw = card.get("network") or payment.get("wallet") or ""
    key = _NETWORK_ALIASES.get(str(raw).strip().lower())
    if key:
        return key

    method = str(payment.get("method") or "").lower()
    if method == "upi":
        return "UPI"
    if method in ("netbanking", "wallet", "emandate", "nach"):
        return "Razorpay"

    code = str(dispute.get("reason_code") or "").upper()
    if code.startswith("RZP"):
        return "Razorpay"
    if _re.fullmatch(r"1[0-3]\.\d+(\.\d+)?", code):
        return "Visa"
    if _re.fullmatch(r"48\d\d", code):
        return "Mastercard"
    if _re.fullmatch(r"[ACFMP]\d\d", code):
        return "American Express"
    return None


def resolve_reason_code(reason_code, network=None, codes=None):
    """reason_code -> the reasoncode.json entry.

    Tries the network's own table first (with Visa's dotted-code convention
    unpacked), then falls back to an exact code match across every network so
    a mis-detected network still resolves. Returns None when the code is not
    in the file - the caller must not guess a requirement list."""
    if not reason_code:
        return None
    codes = codes or load_reason_codes()
    code = str(reason_code).strip()

    def _hit(net, category, entry):
        return {
            "network": net,
            "category": category,
            "code": entry["code"],
            "title": entry.get("title"),
            "description": entry.get("description"),
            "documents": list(entry.get("documents") or []),
            "matched_on": code,
        }

    # 1. Visa dotted code: 13.1 -> Customer Dispute / "1"
    if network == "Visa" or (network is None and "." in code):
        head, _, tail = code.partition(".")
        category = _VISA_CATEGORY_BY_PREFIX.get(head)
        if category and tail:
            for entry in codes.get("Visa", {}).get(category, []):
                if entry["code"] == tail:
                    return _hit("Visa", category, entry)

    # 2. exact code inside the detected network
    if network and network in codes:
        for category, entries in codes[network].items():
            for entry in entries:
                if entry["code"].upper() == code.upper():
                    return _hit(network, category, entry)

    # 3. exact code anywhere (unambiguous only: Visa's 1..8 repeat per category)
    matches = [
        _hit(net, category, entry)
        for net, cats in codes.items()
        for category, entries in cats.items()
        for entry in entries
        if entry["code"].upper() == code.upper()
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def map_document_to_slot(document):
    """One free-text required document -> one Razorpay evidence slot."""
    text = str(document).lower()
    for slot, keywords in DOCUMENT_RULES:
        if any(k in text for k in keywords):
            return slot
    return "others"


def build_requirements(documents):
    """Required documents -> deduped, weighted requirement list, heaviest first.
    Each requirement keeps the source phrases so the agent knows what to fetch."""
    order, grouped = [], {}
    for doc in documents or []:
        slot = map_document_to_slot(doc)
        if slot not in grouped:
            grouped[slot] = []
            order.append(slot)
        if doc not in grouped[slot]:
            grouped[slot].append(doc)
    reqs = [{"type": slot, "weight": SLOT_WEIGHT.get(slot, 1), "documents": grouped[slot]}
            for slot in order]
    reqs.sort(key=lambda r: (-r["weight"], r["type"]))
    return reqs


def extract_submitted_evidence(dispute, requirements):
    """What the dispute entity already carries for each required slot.

    Razorpay returns `evidence` with the slot names as keys (document id lists)
    plus an `others` array. A slot the API did not return stays None - absent
    is not the same as searched-and-missing."""
    evidence_obj = dispute.get("evidence") or {}
    out = {}
    for req in requirements:
        slot = req["type"]
        if slot == "others":
            value = evidence_obj.get("others") or None
        else:
            value = evidence_obj.get(slot) or None
        out[slot] = value
    return out


def score_evidence(requirements, evidence):
    """Weighted coverage of the required slots by what is already on file."""
    max_score = sum(r["weight"] for r in requirements)
    score = sum(r["weight"] for r in requirements if evidence.get(r["type"]))
    missing = [r["type"] for r in requirements if not evidence.get(r["type"])]
    return {
        "coverage_score": score,
        "max_score": max_score,
        "coverage_percent": round(score / max_score * 100) if max_score else 0,
        "missing": missing,
    }


def normalize_case(dispute, webhook_dispute=None, now=None):
    """Build the `case` dict handed to the LLM as dispute context.

    Args:
        dispute: the dispute entity from GET /disputes/:id?expand[]=payment.
        webhook_dispute: the entity off the webhook, used only to fill fields
            the API response is missing (e.g. when the fetch failed).
        now: epoch seconds, for deterministic tests.

    Returns:
        {"dispute": ..., "payment": ..., "requirements": [...],
         "evidence": {...}, "evidence_analysis": {...}}
    """
    now = now if now is not None else _time.time()
    d = {**(webhook_dispute or {}), **(dispute or {})}

    payment = d.get("payment")
    if not isinstance(payment, dict):               # not expanded: id only
        payment = {"id": d.get("payment_id")} if d.get("payment_id") else {}

    network = detect_network(d, payment)
    reason = resolve_reason_code(d.get("reason_code"), network)

    requirements = build_requirements(reason["documents"] if reason else [])
    evidence = extract_submitted_evidence(d, requirements)
    analysis = score_evidence(requirements, evidence)

    respond_by = d.get("respond_by")
    days_remaining = (max(0, _math.ceil((respond_by - now) / 86400))
                      if respond_by else None)

    return {
        "dispute": {
            "id": d.get("id"),
            "status": d.get("status"),
            "phase": d.get("phase"),
            "reason_code": d.get("reason_code"),
            "reason_description": d.get("reason_description") or (reason or {}).get("description"),
            "network": (reason or {}).get("network") or network,
            "category": (reason or {}).get("category"),
            "reason_title": (reason or {}).get("title"),
            "reason_code_resolved": reason is not None,
            "amount": d.get("amount"),                       # paise
            "currency": d.get("currency"),
            "respond_by": respond_by,
            "days_remaining": days_remaining,
            "created_at": d.get("created_at"),
        },
        "payment": {
            "id": payment.get("id"),
            "date": payment.get("created_at"),
            "method": payment.get("method"),
            "email": payment.get("email"),
            "contact": payment.get("contact"),
            "refund_status": (payment.get("refund_status")
                              or ("not_refunded" if payment.get("id") else None)),
        },
        "requirements": requirements,
        "evidence": evidence,
        "evidence_analysis": analysis,
    }
