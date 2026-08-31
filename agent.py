"""Agent orchestrator. Runs the full pipeline for one dispute and emits a case file.

Stage map (see README):
  2 normalize        deterministic
  3 requirements     deterministic
  4 extraction       LLM (or stub)
  5 slot diff        deterministic
  6 score            deterministic
  7 contest / accept LLM for the letter only
  8 gap patcher      LLM (or template)
  9 audit log        deterministic
"""
import json, time, argparse
from pathlib import Path
from core import get_requirements, score, StubExtractor, CODES

ROOT = Path(__file__).parent


class Audit:
    def __init__(self): self.events = []
    def log(self, stage, kind, detail):
        self.events.append({"t": round(time.time(), 3), "stage": stage,
                            "kind": kind, "detail": detail})
    def dump(self): return self.events


# ---------- stage 8: gap patcher (template version; swap for LLM) ----------
PATCH_LIBRARY = {
    "term_and_conditions": {
        "why": "Reason code {code} requires terms showing what the customer agreed to. With no T&C on file the issuer has nothing to weigh against the cardholder's claim, so the dispute is decided on their word alone.",
        "clause": "By placing an order you agree that delivery is deemed complete on handover at the address supplied at checkout, that title and risk pass on delivery, and that our published returns policy governs all post-delivery claims.",
        "where": "Checkout page: mandatory unticked checkbox linking to /terms. Log the accepted version hash and timestamp against the order.",
    },
    "refund_cancellation_policy": {
        "why": "The Razorpay evidence object has a dedicated refund_cancellation_policy slot for {code}. Leaving it null forfeits the argument that the customer had a defined remedy and did not use it.",
        "clause": "Unused items may be returned within 10 days of delivery in original packaging. Refunds are issued to the original payment method within 5-7 working days of receipt. Cancellations accepted until dispatch.",
        "where": "Publish at /refund-policy, link in checkout footer and in the order confirmation email.",
    },
    "shipping_proof": {
        "why": "Delivery confirmation with signature is the highest weighted item for {code}. Without it the merchant cannot rebut non-receipt.",
        "clause": None,
        "where": "Require signature-on-delivery from the courier and pull the POD artifact into the order record automatically via courier webhook.",
    },
    "others:tracking_information": {
        "why": "Tracking history corroborates the POD and is explicitly listed for {code}.",
        "clause": None,
        "where": "Store the AWB and full scan history against the order at dispatch, not just the AWB string.",
    },
    "customer_communication": {
        "why": "{code} asks for customer acknowledgement or proof the customer never raised the issue. With no thread on file neither can be shown.",
        "clause": None,
        "where": "Route all order comms through a logged channel (email or ticketing). Exclude WhatsApp, which Razorpay does not accept as evidence.",
    },
    "proof_of_service": {
        "why": "Service completion records are listed for {code} and are the fallback when POD is unavailable.",
        "clause": None,
        "where": "Have ops mark fulfilment complete with a timestamped note and operator ID.",
    },
    "access_activity_log": {
        "why": "Digital delivery logs support {code} for anything with an online component.",
        "clause": None,
        "where": "Persist delivery webhooks, IP and device fingerprint for 180 days minimum.",
    },
    "others:product_description": {
        "why": "For {code} the listing as it appeared at purchase time is the core rebuttal to 'not as described'.",
        "clause": None,
        "where": "Snapshot the listing (copy + images) at order time and attach it immutably to the order.",
    },
    "others:quality_control": {
        "why": "QC records rebut the 'defective' half of {code}.",
        "clause": None,
        "where": "Keep batch inspection sheets with inspector ID and date, linked to SKU and dispatch batch.",
    },
    "others:no_return_received": {
        "why": "{code} is defeated if you can show the customer kept the goods and raised no RMA.",
        "clause": None,
        "where": "Log RMA state per order and reconcile inbound shipments daily.",
    },
}


def patch_gaps(missing, code):
    out = []
    for slot in missing:
        p = PATCH_LIBRARY.get(slot)
        if not p:
            continue
        out.append({"slot": slot, "why_it_cost_the_dispute": p["why"].format(code=code),
                    "replacement_clause": p["clause"], "capture_at": p["where"]})
    return out


# ---------- stage 7a: representment packet ----------
def build_packet(dispute, verdict, findings):
    req = {i["slot"]: i["doc"] for i in get_requirements(dispute["reason_code"])["items"]}
    cited = [{"required_document": req[s], "evidence_slot": s,
              "proof": next((f.proof for f in findings.findings if f.slot == s), None)}
             for s in verdict.matched]
    summary = (f"Dispute {dispute['reason_code']} contested. "
               f"{len(verdict.matched)} of {len(req)} required documents on file. "
               f"Merchant holds: {', '.join(verdict.matched)}.")[:1000]
    return {"action": "draft",              # never auto-submit
            "dispute_id": dispute["id"],
            "evidence": {"summary": summary, "_cited": cited},
            "_note": "action=draft. Human review required before action=submit."}


# ---------- main pipeline ----------
def run_dispute(dispute, extractor=None, audit=None):
    a = audit or Audit()
    extractor = extractor or StubExtractor()

    a.log(2, "normalize", {"id": dispute["id"], "status": dispute["status"],
                           "phase": dispute["phase"], "amount": dispute["amount"]})
    if dispute["status"] != "open":
        a.log(2, "halt", "status not open, no action permitted")
        return {"halted": True, "audit": a.dump()}

    req = get_requirements(dispute["reason_code"])
    slots = [i["slot"] for i in req["items"]]
    a.log(3, "requirements", {"code": dispute["reason_code"], "n_required": len(slots),
                              "source": CODES["_source"]})

    findings = extractor.extract(dispute, slots)
    a.log(4, "extraction", {"found": [f.slot for f in findings.findings if f.present]})

    v = score(dispute, findings.as_map())
    a.log(5, "slot_diff", {"matched": v.matched, "missing": v.missing})
    a.log(6, "score", {"coverage": v.coverage, "win_probability": v.win_probability,
                       "verdict": v.verdict, "days_to_respond": v.days_to_respond})

    if v.verdict == "contest":
        packet = build_packet(dispute, v, findings)
        a.log(7, "contest_draft", {"cited": len(packet["evidence"]["_cited"])})
    else:
        packet = {"action": "accept_recommended",
                  "dispute_id": dispute["id"],
                  "reason": f"coverage {v.coverage} below threshold. "
                            f"Blocking gaps: {', '.join(v.missing[:3])}.",
                  "cost_if_lost_inr": dispute["amount"] / 100}
        a.log(7, "accept_recommended", {"cost_inr": dispute["amount"] / 100})

    patches = patch_gaps(v.missing, dispute["reason_code"])
    a.log(8, "gap_patch", {"n_patches": len(patches)})
    a.log(9, "done", "case file emitted")

    return {"dispute_id": dispute["id"], "merchant": dispute["merchant"],
            "verdict": v.model_dump(), "packet": packet,
            "gap_patches": patches, "audit": a.dump()}


# ---------- entry B: preflight, no dispute attached ----------
def preflight(dispute, extractor=None):
    """Same stages 3-5, scored per reason code, before anything goes wrong."""
    extractor = extractor or StubExtractor()
    rows = []
    for code in [c for c in CODES if not c.startswith("_")]:
        items = get_requirements(code)["items"]
        probe = dict(dispute); probe["reason_code"] = code
        ext = extractor.extract(probe, [i["slot"] for i in items])
        fm = ext.as_map()
        tw = sum(i["weight"] for i in items)
        gw = sum(i["weight"] for i in items if fm.get(i["slot"]))
        rows.append({"reason_code": code, "name": CODES[code]["name"],
                     "readiness": round(gw / tw, 3),
                     "covered": [i["slot"] for i in items if fm.get(i["slot"])],
                     "gaps": [i["slot"] for i in items if not fm.get(i["slot"])],
                     "exposure_if_disputed_inr": dispute["amount"] / 100})
    return {"merchant": dispute["merchant"], "per_code": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dispute", choices=["dispute", "preflight"])
    ap.add_argument("--index", type=int, default=0)
    a = ap.parse_args()
    rows = json.loads((ROOT / "data/disputes.json").read_text())
    d = rows[a.index]
    out = run_dispute(d) if a.mode == "dispute" else preflight(d)
    print(json.dumps(out, indent=1))