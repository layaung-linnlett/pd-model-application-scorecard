"""Stage 13: score the sealed test pile. Once.

Why now and not before
----------------------
The test pile has sat untouched since `src/03_split.py` wrote it. That was
deliberate: validation was used to make four decisions - class_weight (stage 7),
SMOTE (stage 9), verification_status (stage 10) and the tree models (stage 11).
Every one of those looked at validation and acted on what it saw, so validation
is no longer an innocent bystander. It has been fitted to, indirectly, by me.

Model selection is now finished. Nothing in this script changes the model, the
features or the operating point, so the test pile can be opened. Whatever it
says is the number that gets reported - including if it is worse. That is the
entire point of having kept it sealed, and rerunning until it flatters would
waste a year of the cohort.

What is reported
----------------
The same metric block as `src/07_evaluate.py`, printed for validation and test
side by side so the gap is visible rather than implied. A gap of a point or two
is normal sampling noise on 171,101 rows. A large gap would mean the validation
number was flattered by the four decisions above.

Calibration, briefly
--------------------
The decision rule here is "review the riskiest 10%", which depends only on rank
order, so calibration does not affect it - a model that doubled every
probability would review exactly the same people. But this is called a PD model,
and a PD that fed IFRS 9 expected credit loss or risk-based pricing would need
the absolute probability to be right, not just the ordering. The decile table at
the end is the cheap version of that check: predicted mean risk against observed
default rate, ten buckets. It is not a substitute for a reliability curve and a
Brier score, and it is not claimed as one.
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
MODELS = ROOT / "outputs" / "models"
TARGET = "default_within_12_months"


def add_missing_flags(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in cfg.MISSING_FLAG_COLUMNS:
        X[f"{col}_missing"] = X[col].isna().astype("int8")
    return X


def score(y: np.ndarray, p: np.ndarray) -> dict:
    """Same definitions as src/07_evaluate.py - kept identical on purpose."""
    n, n_def = len(y), int(y.sum())
    k = int(round(n * cfg.REVIEW_CAPACITY))
    order = np.argsort(-p)
    flagged = np.zeros(n, dtype=bool)
    flagged[order[:k]] = True

    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    tn = int((~flagged & (y == 0)).sum())

    auc = roc_auc_score(y, p)
    cum_def = np.cumsum(y[order]) / n_def
    cum_ok = np.cumsum(1 - y[order]) / (n - n_def)
    ks = float((cum_def - cum_ok).max())

    return {"n": n, "n_def": n_def, "k": k, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": tp / (tp + fn), "precision": tp / (tp + fp),
            "auc": auc, "gini": 2 * auc - 1, "ks": ks * 100,
            "lift": (tp / (tp + fn)) / cfg.REVIEW_CAPACITY}


def compare(name: str, v: dict, t: dict) -> None:
    print(f"\n{name}")
    print("-" * 58)
    print(f"{'':<22}{'validation':>13}{'TEST':>13}{'gap':>10}")
    rows = [("defaulters caught", "tp", "{:,}", 0),
            ("recall", "recall", "{:.2%}", 1),
            ("precision", "precision", "{:.2%}", 1),
            ("ROC-AUC", "auc", "{:.4f}", 2),
            ("Gini", "gini", "{:.4f}", 2),
            ("KS", "ks", "{:.1f}", 3),
            ("lift over random", "lift", "{:.2f}x", 4)]
    for label, key, fmt, kind in rows:
        a, b = v[key], t[key]
        if kind == 0:
            gap = f"{b - a:+,}"
        elif kind == 1:
            gap = f"{(b - a) * 100:+.2f}pp"
        elif kind == 4:
            gap = f"{b - a:+.2f}"
        else:
            gap = f"{b - a:+.4f}" if kind == 2 else f"{b - a:+.1f}"
        print(f"{label:<22}{fmt.format(a):>13}{fmt.format(b):>13}{gap:>10}")


def deciles(y: np.ndarray, p: np.ndarray, label: str) -> None:
    """Predicted mean risk vs observed default rate, ten buckets, riskiest first."""
    order = np.argsort(-p)
    print(f"\ncalibration by decile - {label}")
    print(f"  {'decile':>7}{'people':>9}{'predicted':>12}{'observed':>11}"
          f"{'ratio':>8}")
    for i, idx in enumerate(np.array_split(order, 10), start=1):
        pred, obs = p[idx].mean(), y[idx].mean()
        ratio = pred / obs if obs > 0 else float("nan")
        print(f"  {i:>7}{len(idx):>9,}{pred:>11.2%}{obs:>11.2%}{ratio:>8.2f}")
    print("  ratio 1.00 = predicted matches observed; >1 over-predicts risk")


def main() -> None:
    categorical = list(cfg.CATEGORICAL_FEATURES)
    flags = [f"{c}_missing" for c in cfg.MISSING_FLAG_COLUMNS]
    numeric = [c for c in cfg.model_features() if c not in categorical]
    cols = numeric + flags + categorical

    val = add_missing_flags(pd.read_parquet(INTERIM / "val.parquet"))
    test = add_missing_flags(pd.read_parquet(INTERIM / "test.parquet"))
    y_val, y_test = val[TARGET].values, test[TARGET].values

    # OneHotEncoder(handle_unknown="ignore") scores an unseen category as all
    # zeros and says nothing. Say something.
    train_ref = pd.read_parquet(INTERIM / "train.parquet", columns=list(cfg.CATEGORICAL_FEATURES))
    unseen = cfg.unseen_categories(train_ref, test)
    if unseen:
        print("WARNING - categories in test never seen in training:")
        for col, items in unseen.items():
            for k, v in items.items():
                print(f"  {col}='{k}' ({v:,} rows) scored as the reference category")
        print()

    print(f"validation {len(val):,} rows, {y_val.mean():.4%} default rate")
    print(f"TEST       {len(test):,} rows, {y_test.mean():.4%} default rate")
    print(f"operating point: review the riskiest {cfg.REVIEW_CAPACITY:.0%}")

    # --- champion ---------------------------------------------------------
    pipe = joblib.load(MODELS / "baseline_logistic.joblib")
    pv = pipe.predict_proba(val[cols])[:, 1]
    pt = pipe.predict_proba(test[cols])[:, 1]
    lv, lt = score(y_val, pv), score(y_test, pt)
    compare("LOGISTIC REGRESSION (champion)", lv, lt)

    print(f"\n  test confusion matrix at {cfg.REVIEW_CAPACITY:.0%}")
    print(f"  {'':<22}{'DID default':>13}{'did NOT':>12}")
    print(f"  {'REVIEWED':<22}{lt['tp']:>13,}{lt['fp']:>12,}")
    print(f"  {'approved unreviewed':<22}{lt['fn']:>13,}{lt['tn']:>12,}")

    # --- challenger -------------------------------------------------------
    bundle = joblib.load(MODELS / "xgboost_challenger.joblib")
    prep, xgb = bundle["prep"], bundle["model"]
    xv = xgb.predict_proba(prep.transform(val[cols]))[:, 1]
    xt = xgb.predict_proba(prep.transform(test[cols]))[:, 1]
    gv, gt = score(y_val, xv), score(y_test, xt)
    compare("XGBOOST (challenger)", gv, gt)

    print(f"\nchallenger vs champion ON TEST: "
          f"{gt['tp'] - lt['tp']:+,} defaulters, "
          f"Gini {lt['gini']:.4f} -> {gt['gini']:.4f}")

    # --- capacity curve on test ------------------------------------------
    order = np.argsort(-pt)
    n, n_def = len(y_test), int(y_test.sum())
    print("\nTEST capacity curve (champion) - the operating point is a choice:")
    print(f"  {'review':>8}{'caught':>9}{'recall':>9}{'precision':>11}{'lift':>7}")
    for cap in [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]:
        kk = int(round(n * cap))
        c = int(y_test[order[:kk]].sum())
        mark = "  <--" if abs(cap - cfg.REVIEW_CAPACITY) < 1e-9 else ""
        print(f"  {cap:>7.0%}{c:>9,}{c/n_def:>9.1%}{c/kk:>11.2%}"
              f"{(c/n_def)/cap:>6.2f}x{mark}")

    deciles(y_test, pt, "champion, test")
    deciles(y_test, xt, "challenger, test")


if __name__ == "__main__":
    main()
