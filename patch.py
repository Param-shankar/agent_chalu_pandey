"""Counterfactual: apply the gap patches to the merchant, re-run the held-out set,
measure how many disputes flip unwinnable -> winnable and how many rupees that is.

Two scenarios, because they cost very different amounts of merchant effort:
  A  policy-only  : publish T&C + refund policy, wire the checkout consent log.  ~1 day of work.
  B  policy + ops : A, plus signature-on-delivery, tracking capture, logged comms. ~1 sprint.
"""
import json, copy
from pathlib import Path
import pandas as pd
from core import score, label, StubExtractor, get_requirements

ROOT = Path(__file__).parent
POLICY_SLOTS = ["term_and_conditions", "refund_cancellation_policy"]
OPS_SLOTS = ["shipping_proof", "others:tracking_information",
             "customer_communication", "access_activity_log",
             "others:product_description", "others:no_return_received"]


def apply_patch(rows, slots):
    out = []
    for d in rows:
        d2 = copy.deepcopy(d)
        for s in slots:
            if s in d2["_truth"]:
                d2["_truth"][s] = True
        out.append(d2)
    return out


def outcomes(rows, ex):
    res = []
    for d in rows:
        sl = [i["slot"] for i in get_requirements(d["reason_code"])["items"]]
        v = score(d, ex.extract(d, sl).as_map())
        res.append({"id": d["id"], "amount": d["amount"], "tier": d["merchant_tier"],
                    "true": label(d), "pred": v.verdict, "coverage": v.coverage})
    return pd.DataFrame(res)


if __name__ == "__main__":
    rows = json.loads((ROOT / "data/disputes.json").read_text())
    test = rows[-25:]

    scenarios = {
        "baseline":        [],
        "A policy-only":   POLICY_SLOTS,
        "B policy + ops":  POLICY_SLOTS + OPS_SLOTS,
    }

    base = None
    print(f"held-out set: {len(test)} disputes, "
          f"Rs {sum(d['amount'] for d in test)/100:,.0f} total at risk\n")
    print(f"{'scenario':<16} {'winnable':>9} {'flipped':>8} {'Rs recovered':>14}")
    print("-" * 52)
    for name, slots in scenarios.items():
        df = outcomes(apply_patch(test, slots), StubExtractor())
        win = (df["true"] == "contest")
        if base is None:
            base = win.copy()
            flipped, money = 0, 0
        else:
            flipped = int((win & ~base).sum())
            money = df[win & ~base]["amount"].sum() / 100
        print(f"{name:<16} {int(win.sum()):>9} {flipped:>8} {money:>14,.0f}")

    print("\nper-tier flip under scenario A (policy-only):")
    b = outcomes(test, StubExtractor())
    a = outcomes(apply_patch(test, POLICY_SLOTS), StubExtractor())
    m = b[["id", "tier", "amount"]].copy()
    m["before"] = (b["true"] == "contest").values
    m["after"] = (a["true"] == "contest").values
    print(m.groupby("tier").apply(
        lambda g: pd.Series({"n": len(g),
                             "winnable_before": int(g["before"].sum()),
                             "winnable_after": int(g["after"].sum()),
                             "rs_recovered": (g[g["after"] & ~g["before"]]["amount"].sum() / 100)}),
        include_groups=False).to_string())