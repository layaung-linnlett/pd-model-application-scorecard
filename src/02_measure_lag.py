"""Stage 3b: measure the payment-to-delinquency lag instead of assuming it.

The Option B label assumes a borrower reaches 90 days past due roughly 3 months
after their last payment, which is why the cutoff is 9 months (9 + 3 = 12).
That 3 was a guess. This script tests it against the data.

Method
------
Some loans in the cohort were still alive but behind on payments at the snapshot
date, and their status records HOW far behind:

    In Grace Period      (roughly  1-15 days past due)
    Late (16-30 days)
    Late (31-120 days)

For those loans we know the last payment date. Counting months from the last
payment to the snapshot date gives the observed relationship between
"months since last payment" and "how delinquent you officially are".

Caveat: `Late (31-120 days)` is a wide bucket spanning about 1 to 4 months, so
this bounds the lag rather than pinning it exactly.
"""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv"

COLS = ["issue_d", "loan_status", "last_pymnt_d", "last_credit_pull_d"]
LATE_BUCKETS = ["In Grace Period", "Late (16-30 days)", "Late (31-120 days)"]


def main() -> None:
    print("Reading date + status columns from the raw file...")
    parts = []
    for chunk in pd.read_csv(RAW, usecols=COLS, chunksize=400_000, low_memory=False):
        issued = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
        keep = issued.between(cfg.COHORT_START, cfg.COHORT_END + "-31")
        if keep.any():
            sel = chunk.loc[keep].copy()
            sel["issue_dt"] = issued.loc[keep]
            parts.append(sel)
    df = pd.concat(parts, ignore_index=True)

    df["last_pymnt_dt"] = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y", errors="coerce")
    df["pull_dt"] = pd.to_datetime(df["last_credit_pull_d"], format="%b-%Y", errors="coerce")

    snapshot = df["pull_dt"].max()
    print(f"Cohort loans: {len(df):,}   snapshot date: {snapshot:%b %Y}\n")

    def months_to(target: pd.Series) -> pd.Series:
        return ((target.dt.year - df["last_pymnt_dt"].dt.year) * 12
                + (target.dt.month - df["last_pymnt_dt"].dt.month))

    df["months_since_pymnt_global"] = months_to(pd.Series(snapshot, index=df.index))
    df["months_since_pymnt_ownpull"] = months_to(df["pull_dt"])

    print("Months since last payment, for loans still alive but behind at snapshot")
    print("(measured to the global snapshot date, Apr 2019)\n")
    print(f"{'status':<22}{'n':>7}{'p25':>7}{'median':>8}{'p75':>7}{'mean':>7}")
    print("-" * 58)
    for bucket in LATE_BUCKETS:
        s = df.loc[df["loan_status"] == bucket, "months_since_pymnt_global"].dropna()
        if len(s):
            print(f"{bucket:<22}{len(s):>7,}{s.quantile(.25):>7.0f}"
                  f"{s.median():>8.0f}{s.quantile(.75):>7.0f}{s.mean():>7.1f}")

    print("\nSame, measured to each loan's own last credit pull (cross-check)\n")
    print(f"{'status':<22}{'n':>7}{'p25':>7}{'median':>8}{'p75':>7}{'mean':>7}")
    print("-" * 58)
    for bucket in LATE_BUCKETS:
        s = df.loc[df["loan_status"] == bucket, "months_since_pymnt_ownpull"].dropna()
        if len(s):
            print(f"{bucket:<22}{len(s):>7,}{s.quantile(.25):>7.0f}"
                  f"{s.median():>8.0f}{s.quantile(.75):>7.0f}{s.mean():>7.1f}")

    # Reference point: a loan that is paying normally should sit at ~1 month.
    current = df.loc[df["loan_status"] == "Current", "months_since_pymnt_global"].dropna()
    print(f"\nReference - loans still 'Current' (paying normally): "
          f"median {current.median():.0f} months since last payment")

    late31 = df.loc[df["loan_status"] == "Late (31-120 days)",
                    "months_since_pymnt_global"].dropna()
    if len(late31):
        print("\n--- Interpretation ---")
        print(f"'Late (31-120 days)' spans roughly 1-4 months past due.")
        print(f"Observed months since last payment: median {late31.median():.0f}, "
              f"p75 {late31.quantile(.75):.0f}.")
        print(f"The 90-day point sits in the upper part of that bucket, so the")
        print(f"implied lag from last payment to 90 DPD is around "
              f"{late31.quantile(.75):.0f} months.")
        print(f"Current assumption: 3 months (cutoff 9). ")


if __name__ == "__main__":
    main()
