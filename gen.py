"""
Generates synthetic disputes in the Razorpay dispute-entity shape.

Key design point: each merchant has a HIDDEN ground-truth slot map.
That map is rendered into messy unstructured text (policy docs, order notes,
support threads). The agent only ever sees the text. Labels are computed from
the hidden map. This keeps the eval honest: error enters at extraction.
"""
import json, random, time, hashlib
from pathlib import Path

ROOT = Path(__file__).parent
CODES = json.loads((ROOT / "data/reason_codes.json").read_text())

MERCHANT_TIERS = {
    # tier: (prob a given evidence slot is truly present, prob a policy slot is present)
    "mature":    (0.85, 0.90),
    "growing":   (0.60, 0.45),
    "early":     (0.35, 0.15),
}

PRODUCTS = ["aviator sunglasses", "wireless earbuds", "yoga mat", "leather wallet",
            "bluetooth speaker", "running shoes", "phone case", "desk lamp"]


def all_slots(code):
    c = CODES[code]
    return [r["slot"] for r in c["required"]] + [p["slot"] for p in c["policy_required"]]


# ---------- renderers: turn hidden truth into messy text the agent must read ----------

def render_policy_doc(truth, merchant, rng):
    parts = [f"# {merchant} — Store Policies\n"]
    if truth.get("term_and_conditions"):
        parts.append(
            "## Terms of Use\n"
            "By completing checkout you agree to these terms. Orders are dispatched within "
            f"{rng.choice([2,3,5])} business days. Delivery timelines are estimates provided by our "
            "logistics partner. Title and risk pass to the customer on delivery to the address "
            "supplied at checkout.\n")
    else:
        parts.append("## About Us\nWe are a small team passionate about quality products. "
                     "Questions? Write to us any time.\n")
    if truth.get("refund_cancellation_policy"):
        parts.append(
            "## Returns and Refunds\n"
            f"Unused items may be returned within {rng.choice([7,10,14])} days of delivery in original "
            "packaging. Refunds are processed to the original payment method within 5-7 working days "
            "of receiving the returned item. Cancellations are accepted until the order is dispatched.\n")
    else:
        parts.append("## Shipping\nWe ship pan-India. Shipping is free above Rs 999.\n")
    if truth.get("others:product_description"):
        parts.append("## Product Listings\nEvery listing includes full specifications, materials, "
                     "dimensions and at least four photographs shot under neutral lighting.\n")
    parts.append("\nContact: support@" + merchant.lower().replace(" ", "") + ".in\n")
    return "\n".join(parts)


def render_order_record(truth, d, rng):
    lines = [f"ORDER {d['order_id']}  |  {d['product']}  |  Rs {d['amount']//100}",
             f"placed {d['order_date']}  |  payment {d['payment_id']}  |  method card"]
    if truth.get("shipping_proof"):
        lines.append(f"courier: {rng.choice(['Delhivery','BlueDart','Ekart'])} "
                     f"| POD received, signature captured at doorstep, recipient name on file")
    elif truth.get("others:tracking_information"):
        lines.append("courier: Delhivery | AWB issued, last scan 'out for delivery', no POD on file")
    else:
        lines.append("courier: dispatched via local partner, no tracking reference recorded")
    if truth.get("others:tracking_information") and truth.get("shipping_proof"):
        lines.append(f"AWB {rng.randint(10**11, 10**12)} | full scan history available")
    if truth.get("proof_of_service"):
        lines.append("fulfilment: order marked COMPLETE by ops with timestamped completion note")
    if truth.get("access_activity_log"):
        lines.append("system: delivery webhook logged, IP and device fingerprint stored")
    if truth.get("others:quality_control"):
        lines.append("QC: batch inspection sheet attached, inspector ID and date recorded")
    if truth.get("others:no_return_received"):
        lines.append("returns desk: no RMA raised, no inbound shipment against this order")
    return "\n".join(lines)


def render_support_thread(truth, d, rng):
    if truth.get("customer_communication"):
        return (f"[{d['order_date']}] customer: is my order shipped?\n"
                f"[support]: yes, dispatched today, tracking shared.\n"
                f"[+3d] customer: received it, thanks. all good.\n")
    return "(no support tickets or email threads found for this order)"


# ---------- generator ----------

def make_dataset(n=80, seed=7):
    rng = random.Random(seed)
    rows = []
    now = int(time.time())
    for i in range(n):
        code = rng.choice(["13.1", "13.3"])
        tier = rng.choice(["mature", "growing", "early"])
        p_ev, p_pol = MERCHANT_TIERS[tier]
        merchant = f"{rng.choice(['Vayu','Lumen','Kirana','Nova','Saanjh','Trailhead'])} {rng.choice(['Retail','Goods','Co','Store'])}"

        truth = {}
        for r in CODES[code]["required"]:
            truth[r["slot"]] = rng.random() < p_ev
        for p in CODES[code]["policy_required"]:
            truth[p["slot"]] = rng.random() < p_pol

        d = {
            "id": "disp_" + hashlib.md5(f"{seed}{i}".encode()).hexdigest()[:14],
            "entity": "dispute",
            "payment_id": "pay_" + hashlib.md5(f"p{seed}{i}".encode()).hexdigest()[:14],
            "order_id": "order_" + hashlib.md5(f"o{seed}{i}".encode()).hexdigest()[:14],
            "amount": rng.choice([149900, 249900, 89900, 459900, 199900]),
            "currency": "INR",
            "amount_deducted": 0,
            "reason_code": code,
            "phase": "chargeback",
            "status": "open",
            "respond_by": now + rng.randint(3, 25) * 86400,
            "created_at": now - rng.randint(1, 5) * 86400,
            "evidence": {k: None for k in
                         ["shipping_proof", "billing_proof", "cancellation_proof",
                          "customer_communication", "proof_of_service", "explanation_letter",
                          "refund_confirmation", "access_activity_log",
                          "refund_cancellation_policy", "term_and_conditions"]},
            # non-API context the merchant systems hold
            "merchant": merchant,
            "merchant_tier": tier,
            "product": rng.choice(PRODUCTS),
            "order_date": f"2026-0{rng.randint(1,8)}-{rng.randint(10,28)}",
        }
        d["evidence"]["others"] = []
        d["_truth"] = truth  # HIDDEN. stripped before the agent sees it.
        d["sources"] = {
            "policy_doc": render_policy_doc(truth, merchant, rng),
            "order_record": render_order_record(truth, d, rng),
            "support_thread": render_support_thread(truth, d, rng),
        }
        rows.append(d)
    return rows


if __name__ == "__main__":
    rows = make_dataset()
    Path(ROOT / "data/disputes.json").write_text(json.dumps(rows, indent=1))
    tiers = {}
    for r in rows:
        tiers[r["merchant_tier"]] = tiers.get(r["merchant_tier"], 0) + 1
    print(f"generated {len(rows)} disputes -> data/disputes.json")
    print("by tier:", tiers)
    print("by code:", {c: sum(1 for r in rows if r['reason_code'] == c) for c in ['13.1', '13.3']})