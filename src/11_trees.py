"""Stage 11: Random Forest and XGBoost against the logistic baseline.

Why these might help where rebalancing did not
----------------------------------------------
Logistic regression adds weights, so it cannot express "it depends". It has one
weight for loan_amnt and one for annual_inc; it cannot say "loan size matters
more when income is low". A tree can, because each split is conditional on the
one above it. Lending is full of "it depends".

Cost
----
Logistic regression gives a coefficient per feature, so a rejection can be
explained per applicant. Hundreds of trees give a global importance ranking
instead. That is the UK GDPR Article 22 concern in the README, and it is why the
pre-registered bar (commit 7f9acd9) is higher than SMOTE's.

Methodology note
----------------
XGBoost early-stops, which needs a held-out set. Early-stopping on VALIDATION
would let the validation pile influence the model and flatter the comparison,
so 15% is carved out of TRAIN for that. Validation stays clean for all three
models. Random Forest needs no equivalent.

Preprocessing is identical for all three so the model is the only difference.
Scaling is unnecessary for trees but harmless, and keeping it identical keeps
the comparison honest.
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
TARGET = "default_within_12_months"
RANDOM_STATE = 42
BASELINE = 1_663
BANDS = {"reject": 50, "challenger": 250}


def add_missing_flags(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in cfg.MISSING_FLAG_COLUMNS:
        X[f"{col}_missing"] = X[col].isna().astype("int8")
    return X


def make_preprocessor(numeric, flags, categorical):
    return ColumnTransformer([
        ("numeric", Pipeline([("fill", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
        ("flags", "passthrough", flags),
        ("text", Pipeline([
            ("fill", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                     drop="first"))]), categorical),
    ])


def main() -> None:
    train = add_missing_flags(pd.read_parquet(INTERIM / "train.parquet"))
    val = add_missing_flags(pd.read_parquet(INTERIM / "val.parquet"))

    categorical = list(cfg.CATEGORICAL_FEATURES)
    flags = [f"{c}_missing" for c in cfg.MISSING_FLAG_COLUMNS]
    numeric = [c for c in cfg.model_features() if c not in categorical]
    cols = numeric + flags + categorical

    y_train, y_val = train[TARGET], val[TARGET]
    n, total = len(val), int(val[TARGET].sum())
    k = int(round(n * cfg.REVIEW_CAPACITY))

    print(f"train {len(train):,} | validation {n:,} ({total:,} defaulters)")
    print(f"operating point: riskiest {cfg.REVIEW_CAPACITY:.0%} = {k:,}")
    print(f"baseline to beat: {BASELINE:,} defaulters caught")
    print(f"bands: reject under +{BANDS['reject']}, "
          f"challenger +{BANDS['reject']} to +{BANDS['challenger']}, "
          f"reconsider over +{BANDS['challenger']}\n")

    prep = make_preprocessor(numeric, flags, categorical)
    X_train = prep.fit_transform(train[cols])
    X_val = prep.transform(val[cols])
    print(f"encoded to {X_train.shape[1]} columns\n")

    results = {}

    def record(label, p, secs):
        caught = int(y_val.values[np.argsort(-p)[:k]].sum())
        results[label] = {"caught": caught, "auc": roc_auc_score(y_val, p),
                          "secs": secs, "p": p}
        print(f"  {label:<22} {secs:>6.1f}s   caught {caught:,}")

    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)
    record("logistic (baseline)", lr.predict_proba(X_val)[:, 1], time.time() - t0)

    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=50, max_features="sqrt",
        n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    record("random forest", rf.predict_proba(X_val)[:, 1], time.time() - t0)

    # carve an early-stopping set out of TRAIN so validation stays clean
    X_fit, X_stop, y_fit, y_stop = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train,
        random_state=RANDOM_STATE)
    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=2000, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
        eval_metric="auc", early_stopping_rounds=50,
        random_state=RANDOM_STATE, n_jobs=-1)
    xgb.fit(X_fit, y_fit, eval_set=[(X_stop, y_stop)], verbose=False)
    record("xgboost", xgb.predict_proba(X_val)[:, 1], time.time() - t0)
    print(f"  (xgboost stopped at {xgb.best_iteration} trees of 2000)")

    print(f"\n{'model':<24}{'caught':>9}{'recall':>9}{'ROC-AUC':>10}{'Gini':>8}"
          f"{'vs baseline':>13}")
    print("-" * 73)
    for label, r in results.items():
        d = r["caught"] - BASELINE
        delta = "" if label.startswith("logistic") else f"{d:+,}"
        print(f"{label:<24}{r['caught']:>9,}{r['caught']/total:>9.2%}"
              f"{r['auc']:>10.4f}{2*r['auc']-1:>8.4f}{delta:>13}")

    print()
    for label in ["random forest", "xgboost"]:
        d = results[label]["caught"] - BASELINE
        if d < BANDS["reject"]:
            verdict = "REJECT - inside the noise"
        elif d < BANDS["challenger"]:
            verdict = "CHALLENGER - real but modest; keep logistic regression"
        else:
            verdict = "RECONSIDER - large enough to weigh against explainability"
        print(f"{label:<16}{d:+6,} defaulters   {verdict}")


if __name__ == "__main__":
    main()
