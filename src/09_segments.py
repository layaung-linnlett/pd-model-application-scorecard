"""Stage 10: does the model work equally well for everyone?

An overall recall of 26.5% is an AVERAGE. A model can look fine overall and be
much worse for one group. This splits validation into segments and asks, for
each one:

    of the defaulters IN THIS GROUP, how many land in the global riskiest 10%?

That is the right question because the model ranks everyone together - a
borrower is reviewed if they are in the top 10% overall, not the top 10% of
their own segment.

Also reported per segment:
    flag rate    what share of this group gets sent to review at all
    base rate    how often this group actually defaults

Nothing here is changed or fixed. It is a diagnostic.
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

ROOT = Path(__file__).resolve().parents[1]
TARGET = "default_within_12_months"


def add_missing_flags(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in cfg.MISSING_FLAG_COLUMNS:
        X[f"{col}_missing"] = X[col].isna().astype("int8")
    return X


def band(s: pd.Series, edges, labels) -> pd.Series:
    return pd.cut(s, bins=edges, labels=labels, include_lowest=True)


def main() -> None:
    pipe = joblib.load(ROOT / "outputs" / "models" / "baseline_logistic.joblib")
    val = add_missing_flags(pd.read_parquet(ROOT / "data" / "interim" / "val.parquet"))

    categorical = cfg.CATEGORICAL_FEATURES
    flags = [f"{c}_missing" for c in cfg.MISSING_FLAG_COLUMNS]
    numeric = [c for c in cfg.model_features() if c not in categorical]

    y = val[TARGET].values
    p = pipe.predict_proba(val[numeric + flags + categorical])[:, 1]
    n = len(y)
    k = int(round(n * cfg.REVIEW_CAPACITY))
    reviewed = np.zeros(n, dtype=bool)
    reviewed[np.argsort(-p)[:k]] = True

    print(f"validation {n:,} | overall default rate {y.mean():.2%} | "
          f"reviewing the riskiest {cfg.REVIEW_CAPACITY:.0%}")
    print(f"OVERALL RECALL: {y[reviewed].sum() / y.sum():.1%} "
          f"({int(y[reviewed].sum()):,} of {int(y.sum()):,})\n")

    emp = val["emp_length"].fillna("not stated")
    emp = np.where(emp == "not stated", "not stated",
          np.where(emp.isin(["< 1 year", "1 year", "2 years", "3 years"]), "0-3 years",
          np.where(emp == "10+ years", "10+ years", "4-9 years")))

    segments = {
        "home ownership": val["home_ownership"],
        "loan term": val["term"],
        "employment length": pd.Series(emp, index=val.index),
        "annual income": band(val["annual_inc"], [0, 45_000, 65_000, 95_000, np.inf],
                              ["under 45k", "45-65k", "65-95k", "over 95k"]),
        "credit score": band(val["fico_range_low"], [0, 665, 695, 725, np.inf],
                             ["under 665", "665-695", "695-725", "over 725"]),
        "debt-to-income": band(val["dti"], [-1, 12, 19, 26, np.inf],
                               ["under 12", "12-19", "19-26", "over 26"]),
        "loan purpose": val["purpose"].where(
            val["purpose"].isin(val["purpose"].value_counts().head(4).index), "other"),
    }

    for name, seg in segments.items():
        print(f"--- by {name} ---")
        print(f"{'group':<22}{'people':>9}{'defaults':>10}{'base rate':>11}"
              f"{'RECALL':>9}{'flagged':>10}")
        rows = []
        for g in seg.dropna().unique():
            m = (seg == g).values
            defs = int(y[m].sum())
            if defs < 50:
                continue
            rows.append((str(g), int(m.sum()), defs, y[m].mean(),
                         y[m & reviewed].sum() / defs, reviewed[m].mean()))
        for g, cnt, defs, base, rec, flag in sorted(rows, key=lambda r: -r[4]):
            print(f"{g:<22}{cnt:>9,}{defs:>10,}{base:>11.2%}{rec:>9.1%}{flag:>10.1%}")
        if rows:
            best, worst = max(r[4] for r in rows), min(r[4] for r in rows)
            print(f"{'':<22}{'':>9}{'':>10}{'':>11}{'spread':>9} "
                  f"{(best-worst)*100:>.1f} pp\n")


if __name__ == "__main__":
    main()
