"""Eval harness. Runs stages 3-6 over the held-out set and reports honest metrics."""
import json, sys, argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report

from core import get_requirements, score, label, StubExtractor

ROOT = Path(__file__).parent


def run(extractor, rows):
    recs = []
    for d in rows:
        slots = [i["slot"] for i in get_requirements(d["reason_code"])["items"]]
        ext = extractor.extract(d, slots)
        v = score(d, ext.as_map())
        y = label(d)
        recs.append({"id": d["id"], "code": d["reason_code"], "tier": d["merchant_tier"],
                     "amount": d["amount"], "true": y, "pred": v.verdict,
                     "coverage": v.coverage, "win_p": v.win_probability,
                     "n_missing": len(v.missing)})
    return pd.DataFrame(recs)


def report(df):
    y_true = (df["true"] == "contest").astype(int)
    y_pred = (df["pred"] == "contest").astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    # money framing
    fn_cost = df[(y_true == 1) & (y_pred == 0)]["amount"].sum() / 100   # winnable, we told them to give up
    fp_cost = df[(y_true == 0) & (y_pred == 1)]["amount"].sum() / 100   # unwinnable, wasted effort + still lost
    saved  = df[(y_true == 1) & (y_pred == 1)]["amount"].sum() / 100

    print("\n=== CONFUSION MATRIX (positive = contest) ===")
    print(f"  TP {tp:3d}   FP {fp:3d}")
    print(f"  FN {fn:3d}   TN {tn:3d}")
    print(f"\nprecision {p:.3f}   recall {r:.3f}   f1 {f1:.3f}   n={len(df)}")
    print("\n=== MONEY ===")
    print(f"  correctly defended (TP):        Rs {saved:>10,.0f}")
    print(f"  FALSE NEGATIVE cost (winnable   Rs {fn_cost:>10,.0f}   <- worst error")
    print(f"     disputes we said to accept)")
    print(f"  FALSE POSITIVE cost (wasted     Rs {fp_cost:>10,.0f}")
    print(f"     representment effort)")
    print("\n=== BY REASON CODE ===")
    print(df.groupby("code").apply(
        lambda g: pd.Series({"n": len(g), "acc": (g["true"] == g["pred"]).mean().round(3)}),
        include_groups=False))
    print("\n=== BY MERCHANT TIER ===")
    print(df.groupby("tier").apply(
        lambda g: pd.Series({"n": len(g), "acc": (g["true"] == g["pred"]).mean().round(3),
                             "true_contest_rate": (g["true"] == "contest").mean().round(3)}),
        include_groups=False))
    print("\n=== EXCEPTION LIST (every disagreement) ===")
    bad = df[df["true"] != df["pred"]]
    if len(bad) == 0:
        print("  none")
    else:
        print(bad[["id", "code", "tier", "true", "pred", "coverage", "amount"]].to_string(index=False))
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
            "fn_cost_inr": float(fn_cost), "fp_cost_inr": float(fp_cost)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", default="stub", choices=["stub", "llm"])
    ap.add_argument("--holdout", type=int, default=25)
    a = ap.parse_args()

    rows = json.loads((ROOT / "data/disputes.json").read_text())
    train, test = rows[:-a.holdout], rows[-a.holdout:]
    print(f"train {len(train)}  holdout {len(test)}  extractor={a.extractor}")

    ex = StubExtractor() if a.extractor == "stub" else __import__("core").LLMExtractor()
    df = run(ex, test)
    m = report(df)
    (ROOT / "out/metrics.json").write_text(json.dumps(m, indent=1))
    df.to_csv(ROOT / "out/holdout_results.csv", index=False)
    print("\nwrote out/metrics.json, out/holdout_results.csv")