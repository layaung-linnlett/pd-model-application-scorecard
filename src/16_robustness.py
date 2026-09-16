"""Stage 16: how much of this should be believed?

Three questions the project could not answer, all of which a credit risk team
asks and none of which need the model to change.

1. IS THE NUMBER STABLE? Gini 0.3831 is one estimate from one test pile. Quoted
   bare it implies a precision it does not have. Bootstrapping the test set
   gives an interval, and - more usefully - an interval on the CHAMPION-MINUS-
   CHALLENGER difference. "XGBoost is better by 0.044" only means something
   with a spread attached. The difference is resampled in PAIRS, using the same
   rows for both models each time, because they are scored on the same
   borrowers and their errors are correlated.

2. HAS THE POPULATION MOVED? Population Stability Index compares a feature's
   distribution between two piles. It is the standard first alarm in deployed
   credit models: the model has not changed, so if its inputs have, the score
   means something different from what it meant at build time. Train vs test
   here is a weak version of that test - they are random halves of one cohort,
   so anything above the thresholds would be a bug rather than drift - but it
   establishes the baseline a production monitor would compare against.

       PSI < 0.10   stable
       0.10 - 0.25  moderate shift, investigate
       > 0.25       significant shift, do not trust the score

3. IS ANYTHING BEING SCORED BLIND? handle_unknown="ignore" encodes an unseen
   category as all zeros and carries on silently. Stage 13 found one such row
   by accident. This checks on purpose.

Nothing here refits or re-selects. It is measurement of a model already fixed.
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
TARGET = "default_within_12_months"
N_BOOT = 1000
SEED = 42


def add_missing_flags(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in cfg.MISSING_FLAG_COLUMNS:
        X[f"{col}_missing"] = X[col].isna().astype("int8")
    return X


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI between two samples of one feature, using quantile bins of `expected`."""
    e = expected[~pd.isna(expected)]
    a = actual[~pd.isna(actual)]
    if len(e) == 0 or len(a) == 0:
        return float("nan")
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.histogram(e, edges)[0] / len(e)
    a_pct = np.histogram(a, edges)[0] / len(a)
    # floor empty buckets so the log stays finite
    floor = 1e-6
    e_pct = np.clip(e_pct, floor, None)
    a_pct = np.clip(a_pct, floor, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    cats = set(expected.dropna().unique()) | set(actual.dropna().unique())
    e = expected.value_counts(normalize=True)
    a = actual.value_counts(normalize=True)
    floor = 1e-6
    tot = 0.0
    for c in cats:
        ep, ap = max(e.get(c, 0.0), floor), max(a.get(c, 0.0), floor)
        tot += (ap - ep) * np.log(ap / ep)
    return float(tot)


def main() -> None:
    cat = list(cfg.CATEGORICAL_FEATURES)
    flags = [f"{c}_missing" for c in cfg.MISSING_FLAG_COLUMNS]
    num = [c for c in cfg.model_features() if c not in cat]
    cols = num + flags + cat

    train = pd.read_parquet(INTERIM / "train.parquet")
    test = add_missing_flags(pd.read_parquet(INTERIM / "test.parquet"))
    y = test[TARGET].values

    # ---------------- 3. unseen categories (cheapest, do it first) ----------
    print("=" * 62)
    print("UNSEEN CATEGORIES  (silent all-zero encodings)")
    print("=" * 62)
    unseen = cfg.unseen_categories(train, test)
    if not unseen:
        print("  none - every category in test was seen in training\n")
    else:
        for col, items in unseen.items():
            for k, v in items.items():
                print(f"  {col}: '{k}' - {v:,} row(s), encoded as all zeros")
        print("  These borrowers are scored as the reference category with no")
        print("  signal raised. In production this needs an alert, not a shrug.\n")

    # ---------------- 1. bootstrap intervals --------------------------------
    pipe = joblib.load(ROOT / "outputs" / "models" / "baseline_logistic.joblib")
    bundle = joblib.load(ROOT / "outputs" / "models" / "xgboost_challenger.joblib")
    p_champ = pipe.predict_proba(test[cols])[:, 1]
    p_chal = bundle["model"].predict_proba(bundle["prep"].transform(test[cols]))[:, 1]

    print("=" * 62)
    print(f"BOOTSTRAP CONFIDENCE INTERVALS  ({N_BOOT:,} resamples of the test set)")
    print("=" * 62)
    rng = np.random.default_rng(SEED)
    n = len(y)
    gc, gx, gd = [], [], []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        a = 2 * roc_auc_score(yy, p_champ[idx]) - 1
        b = 2 * roc_auc_score(yy, p_chal[idx]) - 1
        gc.append(a); gx.append(b); gd.append(b - a)

    def ci(v):
        return np.percentile(v, 2.5), np.percentile(v, 97.5)

    for label, point, samples in [
            ("logistic (champion)", 2 * roc_auc_score(y, p_champ) - 1, gc),
            ("xgboost (challenger)", 2 * roc_auc_score(y, p_chal) - 1, gx)]:
        lo, hi = ci(samples)
        print(f"  {label:<22} Gini {point:.4f}   95% CI [{lo:.4f}, {hi:.4f}]"
              f"   +/-{(hi-lo)/2:.4f}")

    lo, hi = ci(gd)
    point_d = (2 * roc_auc_score(y, p_chal) - 1) - (2 * roc_auc_score(y, p_champ) - 1)
    print(f"\n  challenger - champion  {point_d:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    if lo > 0:
        print("  Interval excludes zero: the challenger's advantage is real, not noise.")
    else:
        print("  Interval includes zero: the difference is not distinguishable from noise.")
    print("  (paired resamples - both models scored on the same rows each time)\n")

    # ---------------- 2. PSI ------------------------------------------------
    print("=" * 62)
    print("POPULATION STABILITY INDEX  (train vs test)")
    print("=" * 62)
    results = []
    for c in num:
        results.append((c, psi(train[c].values.astype(float), test[c].values.astype(float))))
    for c in cat:
        results.append((c, psi_categorical(train[c], test[c])))
    results = [(c, v) for c, v in results if not np.isnan(v)]
    results.sort(key=lambda r: -r[1])

    worst = results[:5]
    print("  largest 5 of "
          f"{len(results)} features checked:")
    for c, v in worst:
        band = "stable" if v < 0.10 else ("MODERATE" if v < 0.25 else "SIGNIFICANT")
        print(f"    {c:<34}{v:>8.5f}   {band}")
    over = [(c, v) for c, v in results if v >= 0.10]
    print(f"\n  features with PSI >= 0.10: {len(over)}")
    if not over:
        print("  All stable, which is the expected result: train and test are random")
        print("  halves of one cohort, so this is a floor reading rather than a")
        print("  drift test. It is the baseline a production monitor compares new")
        print("  applicants against - and the number that would move first if the")
        print("  applicant population shifted after deployment.")


if __name__ == "__main__":
    main()
