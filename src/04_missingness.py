"""Stage 5: measure missingness, so a missing-data policy can be agreed.

Measured on the TRAINING pile only. The validation and test piles are not
inspected - looking at them to make modelling decisions is a subtle form of
leakage, even when only glancing at summary statistics.

No values are changed here. This script only reports.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "interim" / "train.parquet"
TARGET = "default_within_12_months"

BANDS = [
    (0.90, 1.01, "almost entirely empty (>90%)"),
    (0.50, 0.90, "mostly empty (50-90%)"),
    (0.20, 0.50, "substantially missing (20-50%)"),
    (0.01, 0.20, "some missing (1-20%)"),
    (0.00001, 0.01, "trace missing (<1%)"),
]


def main() -> None:
    df = pd.read_parquet(TRAIN)
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    n = len(X)

    miss = (X.isna().sum() / n).sort_values(ascending=False)
    complete = miss[miss == 0]

    print(f"Training pile: {n:,} rows x {X.shape[1]} features")
    print(f"Fully complete columns: {len(complete)} of {X.shape[1]}\n")

    for lo, hi, label in BANDS:
        cols = miss[(miss > lo) & (miss <= hi)]
        if not len(cols):
            continue
        print(f"--- {label}: {len(cols)} columns ---")
        for col, frac in cols.items():
            print(f"    {col:<34} {frac:>6.1%}")
        print()

    # Does a missing value itself carry signal? Compare the default rate for
    # rows where the column is missing vs present. Only meaningful for columns
    # that are neither almost-always nor almost-never missing.
    print("--- Is 'being missing' itself predictive? ---")
    print("(default rate when missing vs when present; base rate "
          f"{y.mean():.2%})\n")
    print(f"    {'column':<34}{'missing':>9}{'present':>9}{'gap':>8}")
    candidates = miss[(miss > 0.02) & (miss < 0.98)]
    rows = []
    for col in candidates.index:
        m = X[col].isna()
        rate_missing, rate_present = y[m].mean(), y[~m].mean()
        rows.append((col, rate_missing, rate_present, rate_missing - rate_present))
    rows.sort(key=lambda r: abs(r[3]), reverse=True)
    for col, rm, rp, gap in rows[:12]:
        print(f"    {col:<34}{rm:>9.2%}{rp:>9.2%}{gap:>+8.2%}")

    print(f"\n    ...{max(0, len(rows) - 12)} further columns not shown")


# ---------------------------------------------------------------------------
# Follow-up: twelve columns are missing at an identical 46.7%. Are they missing
# on the SAME rows, and if so, what do those rows have in common?
# ---------------------------------------------------------------------------

CLUSTER = [
    "all_util", "inq_last_12m", "total_cu_tl", "open_acc_6m", "open_il_24m",
    "open_act_il", "open_il_12m", "max_bal_bc", "open_rv_12m", "open_rv_24m",
    "inq_fi", "total_bal_il",
]


def check_cluster() -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as cfg

    df = pd.read_parquet(TRAIN)
    mask = df[CLUSTER].isna()

    all_missing = mask.all(axis=1).sum()
    any_missing = mask.any(axis=1).sum()

    print("\n" + "=" * 66)
    print("Are the 12 columns missing on the same rows?\n")
    print(f"  rows where ALL 12 are missing: {all_missing:,}")
    print(f"  rows where ANY is missing:     {any_missing:,}")
    print(f"  -> {'IDENTICAL rows' if all_missing == any_missing else 'NOT identical'}")

    # If they vanish together, something about the row - not the borrower -
    # explains it. Test the obvious candidate: when the loan was issued.
    raw = ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
    print("\nMissingness of `all_util` by month of origination:\n")

    parts = []
    for chunk in pd.read_csv(raw, usecols=["issue_d", "all_util"],
                             chunksize=400_000, low_memory=False):
        issued = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
        keep = issued.between(cfg.COHORT_START, cfg.COHORT_END + "-31")
        if keep.any():
            parts.append(pd.DataFrame({
                "month": issued.loc[keep].dt.to_period("M"),
                "missing": chunk.loc[keep, "all_util"].isna(),
            }))
    obs = pd.concat(parts, ignore_index=True)

    by_month = obs.groupby("month")["missing"].agg(["size", "mean"])
    for month, row in by_month.iterrows():
        bar = "#" * int(row["mean"] * 40)
        print(f"  {month}  {row['size']:>7,}  {row['mean']:>6.1%}  {bar}")


if __name__ == "__main__":
    main()
    check_cluster()
