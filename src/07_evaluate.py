"""Stage 8: evaluate the approved baseline.

Runs on the VALIDATION pile. The test pile stays sealed - SMOTE and tree models
are still untested, so model selection is not finished, and touching test now
would compromise the one honest estimate we get at the end.

Accuracy is deliberately absent. At a 3.70% default rate, a model predicting
"never defaults" scores 96.3% and is worthless. Every metric here is chosen
because it survives class imbalance.

    confusion matrix   the four outcomes, at the stated operating point
    recall             of all defaulters, how many did we catch
    precision          of all our reviews, how many were useful
    ROC-AUC            pick a defaulter and a non-defaulter at random; how
                       often does the model score the defaulter higher
    Gini               = 2*AUC - 1, rescaled so "useless" is 0 not 0.5.
                       The credit-industry convention.
    KS                 the widest gap between how fast defaulters accumulate
                       down the ranked list and how fast non-defaulters do
"""

from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
FIGURES = ROOT / "outputs" / "figures"
TARGET = "default_within_12_months"


def add_missing_flags(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in cfg.MISSING_FLAG_COLUMNS:
        X[f"{col}_missing"] = X[col].isna().astype("int8")
    return X


def main() -> None:
    pipe = joblib.load(ROOT / "outputs" / "models" / "baseline_logistic.joblib")
    val = add_missing_flags(pd.read_parquet(INTERIM / "val.parquet"))

    categorical = cfg.CATEGORICAL_FEATURES
    flags = [f"{c}_missing" for c in cfg.MISSING_FLAG_COLUMNS]
    numeric = [c for c in cfg.model_features() if c not in categorical]

    y = val[TARGET].values
    p = pipe.predict_proba(val[numeric + flags + categorical])[:, 1]
    n, n_def = len(y), int(y.sum())

    # --- operating point --------------------------------------------------
    k = int(round(n * cfg.REVIEW_CAPACITY))
    flagged = np.zeros(n, dtype=bool)
    flagged[np.argsort(-p)[:k]] = True
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    tn = int((~flagged & (y == 0)).sum())
    recall, precision = tp / (tp + fn), tp / (tp + fp)

    print(f"VALIDATION: {n:,} borrowers, {n_def:,} defaulted ({y.mean():.2%})")
    print(f"OPERATING POINT: review the riskiest {cfg.REVIEW_CAPACITY:.0%} = {k:,}")
    print("  (an assumption, not a fact - see cfg.REVIEW_CAPACITY)\n")

    print(f"{'':<24}{'DID default':>13}{'did NOT':>12}")
    print("-" * 49)
    print(f"{'REVIEWED':<24}{tp:>13,}{fp:>12,}")
    print(f"{'approved unreviewed':<24}{fn:>13,}{tn:>12,}")
    print("-" * 49)

    auc = roc_auc_score(y, p)
    order = np.argsort(-p)
    cum_def = np.cumsum(y[order]) / n_def
    cum_ok = np.cumsum(1 - y[order]) / (n - n_def)
    ks_gap = cum_def - cum_ok
    ks_i = int(np.argmax(ks_gap))

    print(f"\n{'recall':<14}{recall:>8.2%}   of defaulters, caught")
    print(f"{'precision':<14}{precision:>8.2%}   of reviews, useful "
          f"(vs {y.mean():.2%} picking at random)")
    print(f"{'ROC-AUC':<14}{auc:>8.4f}   50% = useless, 100% = perfect")
    print(f"{'Gini':<14}{2*auc-1:>8.4f}   = 2*AUC-1, the industry convention")
    print(f"{'KS':<14}{ks_gap[ks_i]*100:>8.1f}   widest separation, at "
          f"{(ks_i+1)/n:.1%} down the list")

    print(f"\nlift over random at the operating point: "
          f"{recall/cfg.REVIEW_CAPACITY:.2f}x")

    print("\nfull capacity curve (the operating point is a choice, so show them all):")
    print(f"  {'review':>8}{'caught':>9}{'recall':>9}{'precision':>11}{'lift':>7}")
    for cap in [0.01, 0.05, 0.10, 0.20, 0.50, 1.00]:
        kk = int(round(n * cap))
        c = int(y[order[:kk]].sum())
        mark = "  <--" if abs(cap - cfg.REVIEW_CAPACITY) < 1e-9 else ""
        print(f"  {cap:>7.0%}{c:>9,}{c/n_def:>9.1%}{c/kk:>11.2%}"
              f"{(c/n_def)/cap:>6.2f}x{mark}")

    # --- figures ----------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    fpr, tpr, _ = roc_curve(y, p)
    ax[0, 0].plot(fpr, tpr, color="#c0392b", lw=2, label=f"model (AUC {auc:.3f})")
    ax[0, 0].plot([0, 1], [0, 1], "--", color="#95a5a6", label="useless (AUC 0.500)")
    ax[0, 0].set_xlabel("share of good borrowers wrongly flagged")
    ax[0, 0].set_ylabel("share of defaulters caught")
    ax[0, 0].set_title("ROC curve", loc="left", weight="bold")
    ax[0, 0].legend(loc="lower right")

    grid = np.arange(1, n + 1) / n
    ax[0, 1].plot(grid * 100, cum_def * 100, color="#c0392b", lw=2, label="model")
    ax[0, 1].plot([0, 100], [0, 100], "--", color="#95a5a6", label="random")
    ax[0, 1].axvline(cfg.REVIEW_CAPACITY * 100, color="#2c3e50", lw=1)
    ax[0, 1].annotate(f"{recall:.1%} of defaulters\ncaught here",
                      xy=(cfg.REVIEW_CAPACITY * 100, recall * 100),
                      xytext=(30, 22), fontsize=9,
                      arrowprops=dict(arrowstyle="->", color="#2c3e50"))
    ax[0, 1].set_xlabel("% of applicants reviewed")
    ax[0, 1].set_ylabel("% of defaulters caught")
    ax[0, 1].set_title("How many you catch for how many you review",
                       loc="left", weight="bold")
    ax[0, 1].legend(loc="lower right")

    ax[1, 0].plot(grid * 100, cum_def * 100, color="#c0392b", lw=2, label="defaulters")
    ax[1, 0].plot(grid * 100, cum_ok * 100, color="#27ae60", lw=2, label="non-defaulters")
    ax[1, 0].vlines((ks_i + 1) / n * 100, cum_ok[ks_i] * 100, cum_def[ks_i] * 100,
                    color="#2c3e50", lw=2)
    ax[1, 0].annotate(f"KS = {ks_gap[ks_i]*100:.1f}",
                      xy=((ks_i + 1) / n * 100, (cum_def[ks_i] + cum_ok[ks_i]) / 2 * 100),
                      xytext=(48, 38), fontsize=10, weight="bold")
    ax[1, 0].set_xlabel("% of applicants read, riskiest first")
    ax[1, 0].set_ylabel("% of that group collected")
    ax[1, 0].set_title("KS: defaulters pile up faster", loc="left", weight="bold")
    ax[1, 0].legend(loc="lower right")

    bins = np.linspace(0, max(p.max(), 0.35), 60)
    ax[1, 1].hist(p[y == 0], bins=bins, color="#27ae60", alpha=.65,
                  density=True, label="did not default")
    ax[1, 1].hist(p[y == 1], bins=bins, color="#c0392b", alpha=.65,
                  density=True, label="defaulted")
    ax[1, 1].set_xlabel("predicted probability of default")
    ax[1, 1].set_ylabel("density")
    ax[1, 1].set_title("The two groups overlap heavily - this is the hard part",
                       loc="left", weight="bold")
    ax[1, 1].legend()

    fig.suptitle(f"Baseline logistic regression — validation set  "
                 f"(AUC {auc:.3f} · Gini {2*auc-1:.3f} · KS {ks_gap[ks_i]*100:.1f})",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    out = FIGURES / "baseline_evaluation.png"
    fig.savefig(out, dpi=140)
    print(f"\nsaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
