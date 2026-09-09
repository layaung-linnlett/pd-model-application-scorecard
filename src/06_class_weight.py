"""Stage 7: does class_weight='balanced' help?

Only ONE thing changes between the two models:

    LogisticRegression(max_iter=1000, random_state=42)
    LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")

Same features, same split, same preprocessing. Any difference is the weighting.

What 'balanced' does
--------------------
Without it, every borrower counts equally. At 26:1 the model learns that guessing
"safe" is nearly always right, and it flags almost nobody.

With it, each defaulter counts ~26x more, so missing one costs 26x a false alarm.

What we measure
---------------
Not accuracy, which is meaningless at a 3.70% base rate (predicting "never
defaults" scores 96.3%). The business question is:

    "We can hand-review the riskiest 5% of applicants. How many real
     defaulters does that catch?"

That depends on the ORDER the model puts borrowers in, not on how worried it is
about them in absolute terms. Watch whether the two differ.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
TARGET = "default_within_12_months"
CAPACITY = 0.05          # share of applicants a lender can hand-review
RANDOM_STATE = 42


def add_missing_flags(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in cfg.MISSING_FLAG_COLUMNS:
        X[f"{col}_missing"] = X[col].isna().astype("int8")
    return X


def build(numeric, flags, categorical, class_weight):
    pre = ColumnTransformer([
        ("numeric", Pipeline([("fill", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
        ("flags", "passthrough", flags),
        ("text", Pipeline([
            ("fill", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                     drop="first"))]), categorical),
    ])
    return Pipeline([
        ("prep", pre),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                     class_weight=class_weight)),
    ])


def main() -> None:
    train = add_missing_flags(pd.read_parquet(INTERIM / "train.parquet"))
    val = add_missing_flags(pd.read_parquet(INTERIM / "val.parquet"))
    y_train, y_val = train[TARGET], val[TARGET]

    categorical = cfg.CATEGORICAL_FEATURES
    flags = [f"{c}_missing" for c in cfg.MISSING_FLAG_COLUMNS]
    numeric = [c for c in cfg.model_features() if c not in categorical]
    cols = numeric + flags + categorical

    k = int(round(len(val) * CAPACITY))
    total_defaulters = int(y_val.sum())
    print(f"validation: {len(val):,} borrowers, {total_defaulters:,} of them defaulted")
    print(f"review capacity {CAPACITY:.0%} -> the riskiest {k:,} get checked by hand\n")

    results = {}
    for label, weight in [("baseline (no weighting)", None),
                          ("class_weight='balanced'", "balanced")]:
        pipe = build(numeric, flags, categorical, weight)
        pipe.fit(train[cols], y_train)
        p = pipe.predict_proba(val[cols])[:, 1]
        top = np.argsort(-p)[:k]
        results[label] = {
            "p": p,
            "caught": int(y_val.values[top].sum()),
            "auc": roc_auc_score(y_val, p),
            "flagged_at_half": int((p >= 0.5).sum()),
            "mean_p": p.mean(),
        }

    print("=" * 78)
    print("WHAT CHANGED: how worried the model is\n")
    print(f"{'model':<28}{'flagged at 0.5':>16}{'average probability':>22}")
    print("-" * 78)
    for label, r in results.items():
        print(f"{label:<28}{r['flagged_at_half']:>16,}{r['mean_p']:>21.2%}")

    print("\n" + "=" * 78)
    print("WHAT MATTERS: who it puts at the top\n")
    print(f"{'model':<28}{'defaulters caught':>19}{'recall':>10}{'ROC-AUC':>10}")
    print("-" * 78)
    for label, r in results.items():
        print(f"{label:<28}{r['caught']:>19,}{r['caught']/total_defaulters:>10.2%}"
              f"{r['auc']:>10.4f}")

    a = results["baseline (no weighting)"]
    b = results["class_weight='balanced'"]
    diff = b["caught"] - a["caught"]
    noise = np.sqrt(a["caught"])
    print(f"\ndifference: {diff:+,} defaulters caught "
          f"({diff / a['caught'] * 100:+.2f}% relative)")
    print(f"noise on a count of ~{a['caught']:,} is roughly +/-{noise:.0f}")
    print(f"-> {'WITHIN' if abs(diff) < noise else 'LARGER THAN'} noise")

    # Did the ORDER change at all?
    order_a = np.argsort(-a["p"])[:k]
    order_b = np.argsort(-b["p"])[:k]
    overlap = len(set(order_a) & set(order_b))
    print(f"\nof the {k:,} borrowers each model sends to review, "
          f"{overlap:,} are the SAME people ({overlap/k:.1%} overlap)")
    print(f"rank correlation between the two models' scores: "
          f"{pd.Series(a['p']).corr(pd.Series(b['p']), method='spearman'):.4f}")


if __name__ == "__main__":
    main()
