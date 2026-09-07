"""Stage 6: logistic regression baseline.

A plain logistic regression, with NO class weighting and NO resampling. That is
deliberate: it is the honest starting point, and the point against which every
later change has to justify itself.

Pipeline
--------
    1. add the 12 `_missing` flags        (deterministic per row - no fitting,
                                           so it cannot leak)
    2. numeric  -> median fill -> scale   (median and scale learned on TRAIN only)
       flags    -> passed through untouched (already 0/1)
       text     -> "Unknown" fill -> one-hot encode
    3. logistic regression

Everything after step 1 sits inside a scikit-learn Pipeline, so the fill values
and scaling factors are fitted on the training pile and merely APPLIED to
validation. Fitting them on all the data would leak.
"""

from pathlib import Path
import sys

import joblib
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
MODELS = ROOT / "outputs" / "models"
TARGET = "default_within_12_months"
RANDOM_STATE = 42


def add_missing_flags(X: pd.DataFrame) -> pd.DataFrame:
    """Add a 0/1 column recording whether each flagged feature was blank.

    Deterministic row by row - nothing is learned from the data - so this is
    safe to apply to each pile independently.
    """
    X = X.copy()
    for col in cfg.MISSING_FLAG_COLUMNS:
        X[f"{col}_missing"] = X[col].isna().astype("int8")
    return X


def build_pipeline(numeric: list[str], flags: list[str],
                   categorical: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("numeric", Pipeline([
            ("fill", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("flags", "passthrough", flags),
        ("text", Pipeline([
            ("fill", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                     drop="first")),
        ]), categorical),
    ])
    return Pipeline([
        ("prep", pre),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])


def main() -> None:
    train = add_missing_flags(pd.read_parquet(INTERIM / "train.parquet"))
    val = add_missing_flags(pd.read_parquet(INTERIM / "val.parquet"))

    categorical = cfg.CATEGORICAL_FEATURES
    flags = [f"{c}_missing" for c in cfg.MISSING_FLAG_COLUMNS]
    numeric = [c for c in cfg.model_features() if c not in categorical]

    X_train, y_train = train[numeric + flags + categorical], train[TARGET]
    X_val, y_val = val[numeric + flags + categorical], val[TARGET]

    print(f"train {len(X_train):,} rows | val {len(X_val):,} rows")
    print(f"columns in: {len(numeric)} numeric + {len(flags)} flags "
          f"+ {len(categorical)} text\n")

    pipe = build_pipeline(numeric, flags, categorical)
    print("fitting...")
    pipe.fit(X_train, y_train)

    model = pipe.named_steps["model"]
    names = pipe.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    print(f"converged in {model.n_iter_[0]} iterations")
    print(f"columns after one-hot encoding: {len(names)}")

    p_val = pipe.predict_proba(X_val)[:, 1]
    print(f"\nROC-AUC on validation: {roc_auc_score(y_val, p_val):.4f}")
    print("(we interpret this properly in Stage 7 - it is here only to confirm "
          "the model learned something)")

    # How many does it actually FLAG at the default 0.5 threshold?
    flagged = int((p_val >= 0.5).sum())
    print(f"\nBorrowers predicted to default (probability >= 0.5): "
          f"{flagged:,} of {len(X_val):,}")
    print(f"Borrowers who actually defaulted:                     "
          f"{int(y_val.sum()):,}")
    print(f"Highest probability the model assigned to anyone:     {p_val.max():.1%}")

    coefs = (pd.DataFrame({"feature": names, "coefficient": model.coef_[0]})
             .assign(abs_coef=lambda d: d.coefficient.abs())
             .sort_values("abs_coef", ascending=False)
             .drop(columns="abs_coef"))
    out = ROOT / "outputs" / "baseline_coefficients.csv"
    coefs.to_csv(out, index=False)

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS / "baseline_logistic.joblib")
    print(f"\nsaved {out.relative_to(ROOT)}")
    print(f"saved {(MODELS / 'baseline_logistic.joblib').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
