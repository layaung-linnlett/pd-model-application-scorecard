"""Stage 12: explain individual XGBoost predictions with SHAP.

SHAP splits a single prediction's distance from average among the features
that caused it, using the Shapley value from game theory - the same rule you
would use to split a shared taxi fare by who was in the car for which leg.

    average borrower          3.7%
    this borrower            18.0%
    gap to explain          +14.3 points  <- SHAP divides this among 60 features

Two honest limits, both reported here rather than buried:

  1. SHAP describes what the MODEL did, not why the borrower is risky. XGBoost
     never computed these numbers; they are reconstructed afterwards by probing
     the model. A logistic regression coefficient IS the logic. This is a
     description of behaviour.

  2. With correlated features the split is not unique. This project has features
     correlated at 0.92; SHAP's division of credit between such a pair is partly
     arbitrary, so an explanation could name either one.

Whether a post-hoc attribution satisfies a right-to-explanation obligation
(UK GDPR Art. 22) is a governance question, not a technical one.
"""

from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from glossary import MEANING

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
FIGURES = ROOT / "outputs" / "figures"
MODELS = ROOT / "outputs" / "models"
TARGET = "default_within_12_months"
RANDOM_STATE = 42


def add_missing_flags(X):
    X = X.copy()
    for col in cfg.MISSING_FLAG_COLUMNS:
        X[f"{col}_missing"] = X[col].isna().astype("int8")
    return X


def pretty(name):
    """Turn an encoded column name into something readable."""
    base = name.split("__", 1)[-1]
    if "_" in base and base.split("_")[0] in ("term", "purpose", "home", "emp",
                                              "application"):
        return base.replace("_", " ")
    return base


def main():
    train = add_missing_flags(pd.read_parquet(INTERIM / "train.parquet"))
    val = add_missing_flags(pd.read_parquet(INTERIM / "val.parquet"))
    categorical = list(cfg.CATEGORICAL_FEATURES)
    flags = [f"{c}_missing" for c in cfg.MISSING_FLAG_COLUMNS]
    numeric = [c for c in cfg.model_features() if c not in categorical]
    cols = numeric + flags + categorical
    y_train, y_val = train[TARGET], val[TARGET]

    prep = ColumnTransformer([
        ("numeric", Pipeline([("fill", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
        ("flags", "passthrough", flags),
        ("text", Pipeline([
            ("fill", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                     drop="first"))]), categorical),
    ])
    X_train = prep.fit_transform(train[cols])
    X_val = prep.transform(val[cols])
    names = [pretty(n) for n in prep.get_feature_names_out()]

    X_fit, X_stop, y_fit, y_stop = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE)
    xgb = XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=5,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                        eval_metric="auc", early_stopping_rounds=50,
                        random_state=RANDOM_STATE, n_jobs=-1)
    xgb.fit(X_fit, y_fit, eval_set=[(X_stop, y_stop)], verbose=False)
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"prep": prep, "model": xgb, "names": names},
                MODELS / "xgboost_challenger.joblib")
    p = xgb.predict_proba(X_val)[:, 1]
    print(f"xgboost refitted ({xgb.best_iteration} trees), "
          f"mean predicted risk {p.mean():.2%}\n")

    explainer = shap.TreeExplainer(xgb)
    sample = np.random.default_rng(RANDOM_STATE).choice(len(X_val), 5000, replace=False)
    sv_sample = explainer.shap_values(X_val[sample])

    # --- pick borrowers to explain -----------------------------------------
    order = np.argsort(-p)
    riskiest_defaulted = next(i for i in order if y_val.iloc[i] == 1)
    riskiest_did_not = next(i for i in order if y_val.iloc[i] == 0)
    safest = order[-1]

    for title, idx in [("HIGHEST-RISK BORROWER WHO DID DEFAULT", riskiest_defaulted),
                       ("HIGHEST-RISK BORROWER WHO DID *NOT* DEFAULT", riskiest_did_not),
                       ("LOWEST-RISK BORROWER IN THE WHOLE SET", safest)]:
        sv = explainer.shap_values(X_val[idx:idx+1])[0]
        base = explainer.expected_value
        print("=" * 74)
        print(f"{title}")
        print(f"  model says {p[idx]:.1%} risk   |   actually defaulted: "
              f"{'YES' if y_val.iloc[idx] else 'no'}")
        print(f"  average borrower is {1/(1+np.exp(-base)):.1%}\n")
        top = np.argsort(-np.abs(sv))[:8]
        print(f"  {'feature':<26}{'their value':>14}{'pushes risk':>13}")
        print("  " + "-" * 55)
        for j in top:
            raw = val[cols].iloc[idx].get(names[j], "")
            raw = f"{raw:,.0f}" if isinstance(raw, (int, float, np.floating)) \
                and not pd.isna(raw) else str(raw)[:13]
            arrow = "UP  " if sv[j] > 0 else "DOWN"
            print(f"  {names[j]:<26}{raw:>14}{arrow:>9} {abs(sv[j]):.2f}")
        print()

    # --- figures ------------------------------------------------------------
    plt.figure()
    shap.summary_plot(sv_sample, X_val[sample], feature_names=names,
                      max_display=15, show=False, plot_size=(11, 7))
    plt.title("Which features drive XGBoost's predictions, and in which direction",
              fontsize=12, loc="left", weight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_summary.png", dpi=140, bbox_inches="tight")
    plt.close()

    ex = shap.Explanation(
        values=explainer.shap_values(X_val[riskiest_defaulted:riskiest_defaulted+1])[0],
        base_values=explainer.expected_value,
        data=X_val[riskiest_defaulted], feature_names=names)
    plt.figure()
    shap.plots.waterfall(ex, max_display=12, show=False)
    plt.title(f"Why this borrower scored {p[riskiest_defaulted]:.0%}",
              fontsize=12, loc="left", weight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_one_borrower.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"saved {(FIGURES / 'shap_summary.png').relative_to(ROOT)}")
    print(f"saved {(FIGURES / 'shap_one_borrower.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
