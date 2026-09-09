"""Stage 9: does SMOTE earn its place?

SMOTE invents synthetic defaulters until the classes balance, so the model
trains partly on borrowers who never applied for a loan.

Two things matter about HOW it is applied here:

  1. Inside an imblearn Pipeline, so SMOTE runs only during `fit`. It must never
     touch validation - inventing validation borrowers would mean scoring the
     model on people who do not exist.
  2. AFTER preprocessing, because SMOTE interpolates between rows and needs
     numeric input. Median fills and scaling are still learned on train only.

Pre-registered bar (commit fb8f998, set before this was run):
    keep SMOTE only if it catches >= 1,728 defaulters at 10% review capacity,
    i.e. +50 on the baseline's 1,678.
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
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
BAR = 1_728          # pre-registered
BASELINE = 1_678
RANDOM_STATE = 42


def add_missing_flags(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in cfg.MISSING_FLAG_COLUMNS:
        X[f"{col}_missing"] = X[col].isna().astype("int8")
    return X


def preprocessor(numeric, flags, categorical):
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
    y_train, y_val = train[TARGET], val[TARGET]

    categorical = cfg.CATEGORICAL_FEATURES
    flags = [f"{c}_missing" for c in cfg.MISSING_FLAG_COLUMNS]
    numeric = [c for c in cfg.model_features() if c not in categorical]
    cols = numeric + flags + categorical

    n = len(val)
    k = int(round(n * cfg.REVIEW_CAPACITY))
    total = int(y_val.sum())
    print(f"train {len(train):,} rows: {int(y_train.sum()):,} defaulters vs "
          f"{int((1-y_train).sum()):,} non-defaulters "
          f"({(1-y_train).sum()/y_train.sum():.1f}:1)")
    print(f"validation {n:,} rows, {total:,} defaulters")
    print(f"operating point: review the riskiest {cfg.REVIEW_CAPACITY:.0%} = {k:,}\n")
    print(f"PRE-REGISTERED BAR: SMOTE must catch >= {BAR:,} "
          f"(baseline {BASELINE:,}, +50)\n")

    results = {}
    for label, use_smote in [("baseline", False), ("SMOTE", True)]:
        steps = [("prep", preprocessor(numeric, flags, categorical))]
        if use_smote:
            steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
        steps.append(("model", LogisticRegression(max_iter=1000,
                                                  random_state=RANDOM_STATE)))
        pipe = ImbPipeline(steps)

        t0 = time.time()
        pipe.fit(train[cols], y_train)
        secs = time.time() - t0

        p = pipe.predict_proba(val[cols])[:, 1]
        caught = int(y_val.values[np.argsort(-p)[:k]].sum())
        results[label] = {"p": p, "caught": caught,
                          "auc": roc_auc_score(y_val, p), "secs": secs}
        print(f"  {label:<10} fitted in {secs:>6.1f}s   caught {caught:,}")

    a, b = results["baseline"], results["SMOTE"]
    print(f"\n{'model':<12}{'caught':>9}{'recall':>9}{'ROC-AUC':>10}{'fit time':>11}")
    print("-" * 51)
    for label, r in results.items():
        print(f"{label:<12}{r['caught']:>9,}{r['caught']/total:>9.2%}"
              f"{r['auc']:>10.4f}{r['secs']:>10.1f}s")

    diff = b["caught"] - a["caught"]
    print(f"\ndifference: {diff:+,} defaulters ({diff/a['caught']*100:+.2f}% relative)")
    print(f"bar was:    +{BAR - BASELINE} defaulters (+3%)")
    print(f"noise on a count of ~{a['caught']:,} is roughly +/-{np.sqrt(a['caught']):.0f}")
    print(f"\nVERDICT: {'KEEP' if b['caught'] >= BAR else 'REJECT'} SMOTE")

    same = len(set(np.argsort(-a["p"])[:k]) & set(np.argsort(-b["p"])[:k]))
    print(f"\nof the {k:,} sent to review by each model, {same:,} are the same "
          f"people ({same/k:.1%})")
    print(f"rank correlation: "
          f"{pd.Series(a['p']).corr(pd.Series(b['p']), method='spearman'):.4f}")
    print(f"\nmean predicted probability: baseline {a['p'].mean():.2%}, "
          f"SMOTE {b['p'].mean():.2%}")


if __name__ == "__main__":
    main()
