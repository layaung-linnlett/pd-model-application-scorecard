"""Feature ablation: does a feature earn its place?

Refits the approved pipeline with a feature removed and compares against the
baseline at the stated operating point. Used three times in this project:

    credit_history_months   dropped (age proxy, no measurable gain)
    verification_status     see docs / PROGRESS.md
    mo_sin_old_*            dropped with credit_history_months

Every comparison judges "defaulters caught at cfg.REVIEW_CAPACITY", not
accuracy and not ROC-AUC, because that is the number the business decision
actually turns on. Bars are pre-registered in git before the run.

Usage:
    ./.venv/bin/python src/10_ablation.py verification_status
    ./.venv/bin/python src/10_ablation.py annual_inc dti
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
RANDOM_STATE = 42


def add_missing_flags(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in cfg.MISSING_FLAG_COLUMNS:
        X[f"{col}_missing"] = X[col].isna().astype("int8")
    return X


def fit_and_score(train, val, y_train, y_val, numeric, flags, categorical, k):
    pre = ColumnTransformer([
        ("numeric", Pipeline([("fill", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
        ("flags", "passthrough", flags),
        ("text", Pipeline([
            ("fill", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                     drop="first"))]), categorical),
    ]) if categorical else ColumnTransformer([
        ("numeric", Pipeline([("fill", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
        ("flags", "passthrough", flags),
    ])
    pipe = Pipeline([("prep", pre),
                     ("model", LogisticRegression(max_iter=1000,
                                                  random_state=RANDOM_STATE))])
    cols = numeric + flags + categorical
    pipe.fit(train[cols], y_train)
    p = pipe.predict_proba(val[cols])[:, 1]
    caught = int(y_val.values[np.argsort(-p)[:k]].sum())
    return caught, roc_auc_score(y_val, p)


def main() -> None:
    drop = sys.argv[1:]
    if not drop:
        print(__doc__)
        sys.exit(1)

    train = add_missing_flags(pd.read_parquet(INTERIM / "train.parquet"))
    val = add_missing_flags(pd.read_parquet(INTERIM / "val.parquet"))
    y_train, y_val = train[TARGET], val[TARGET]

    categorical = list(cfg.CATEGORICAL_FEATURES)
    flags = [f"{c}_missing" for c in cfg.MISSING_FLAG_COLUMNS]
    numeric = [c for c in cfg.model_features() if c not in categorical]

    unknown = [c for c in drop if c not in numeric + categorical]
    if unknown:
        print(f"not in the model: {unknown}")
        sys.exit(1)

    k = int(round(len(val) * cfg.REVIEW_CAPACITY))
    total = int(y_val.sum())
    print(f"validation {len(val):,} | {total:,} defaulters | "
          f"reviewing the riskiest {cfg.REVIEW_CAPACITY:.0%} = {k:,}\n")

    base_caught, base_auc = fit_and_score(train, val, y_train, y_val,
                                          numeric, flags, categorical, k)
    print(f"{'model':<34}{'caught':>9}{'recall':>9}{'ROC-AUC':>10}")
    print("-" * 62)
    print(f"{'baseline (all features)':<34}{base_caught:>9,}"
          f"{base_caught/total:>9.2%}{base_auc:>10.4f}")

    kept_num = [c for c in numeric if c not in drop]
    kept_cat = [c for c in categorical if c not in drop]
    caught, auc = fit_and_score(train, val, y_train, y_val,
                                kept_num, flags, kept_cat, k)
    label = "without " + ", ".join(drop)
    print(f"{label[:33]:<34}{caught:>9,}{caught/total:>9.2%}{auc:>10.4f}")

    diff = caught - base_caught
    print(f"\ncost of removing: {-diff:+,} defaulters "
          f"({-diff/base_caught*100:+.2f}% relative)")
    print(f"sampling noise on ~{base_caught:,} is roughly "
          f"+/-{np.sqrt(base_caught):.0f}")
    print(f"ROC-AUC change: {auc - base_auc:+.4f}")
    verdict = "WITHIN noise - removal is effectively free" if abs(diff) < 41 \
        else "OUTSIDE noise - the feature is carrying real weight"
    print(f"\n{verdict}")


if __name__ == "__main__":
    main()
