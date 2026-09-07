"""Stage 3: build the modelling dataset.

Three jobs:
  1. Keep only loans originated 2015-01 .. 2016-12.
  2. Construct the target, `default_within_12_months`, using the Option B rule.
  3. Apply the column policy from config.py and save to data/interim/.

Option B rule
-------------
The file records one snapshot status per loan (as at Apr 2019), not a month-by-month
payment history, so "reached 90+ days past due within 12 months" is not directly
observable. We reconstruct it:

    a borrower who stops paying reaches 90 DPD after some lag,
    so a last payment within (12 - lag) months implies 90 DPD by month 12.

    default_within_12_months = 1  if the loan ended in Charged Off / Default
                                  AND months(issue_d -> last_pymnt_d) <= CUTOFF
                                  (a missing last_pymnt_d means they never paid
                                   at all, which also counts as 1)
                             = 0  otherwise

The lag was originally guessed at 3 months (cutoff 9). `src/02_measure_lag.py`
measured it against loans that were delinquent at the snapshot and found the
90-day point sits at roughly 4 months since last payment, so the cutoff is 8.

Cutoffs 9 and 12 are still reported as a sensitivity check, since the measurement
brackets the lag at roughly 3-4 months rather than pinning it exactly. All three
numbers belong in the README.
"""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
OUT = ROOT / "data" / "interim" / "cohort_2015_2016.parquet"

DEAD_STATUSES = {"Charged Off", "Default"}
CUTOFF_MONTHS = 8            # 12 - 4, using the measured lag (see 02_measure_lag.py)
SENSITIVITY_CUTOFFS = [9, 12]   # reported alongside; not used for the label

READ_COLS = cfg.LABEL_SOURCE + cfg.FEATURES


def months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    """Whole months from `start` to `end`. NaN if either date is missing."""
    return (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)


def main() -> None:
    print(f"Reading {RAW.name} ({len(READ_COLS)} of 151 columns)...")

    kept = []
    rows_scanned = 0
    for chunk in pd.read_csv(RAW, usecols=READ_COLS, chunksize=250_000, low_memory=False):
        rows_scanned += len(chunk)
        issued = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
        in_cohort = issued.between(cfg.COHORT_START, cfg.COHORT_END + "-31")
        if in_cohort.any():
            sel = chunk.loc[in_cohort].copy()
            sel["issue_dt"] = issued.loc[in_cohort]
            kept.append(sel)

    df = pd.concat(kept, ignore_index=True)
    print(f"  scanned {rows_scanned:,} rows -> kept {len(df):,} in cohort\n")

    # --- Build the label ---------------------------------------------------
    # --- Derived feature: how long they have held credit ------------------
    # A date string is useless to a model; the LENGTH of credit history is not.
    # This is the only legitimate use of issue_d - it is discarded straight
    # after, and never becomes a feature itself.
    earliest = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["credit_history_months"] = months_between(earliest, df["issue_dt"])

    df["last_pymnt_dt"] = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y", errors="coerce")
    df["months_to_last_pymnt"] = months_between(df["issue_dt"], df["last_pymnt_dt"])

    died = df["loan_status"].isin(DEAD_STATUSES)
    never_paid = df["months_to_last_pymnt"].isna()

    def label_at(cutoff: int) -> pd.Series:
        within = never_paid | (df["months_to_last_pymnt"] <= cutoff)
        return (died & within).astype("int8")

    df["default_within_12_months"] = label_at(CUTOFF_MONTHS)

    # --- Report ------------------------------------------------------------
    print("loan_status in cohort:")
    for status, n in df["loan_status"].value_counts().items():
        flag = "  <-- counts as died" if status in DEAD_STATUSES else ""
        print(f"  {status:<45} {n:>8,}{flag}")

    n = len(df)
    n_died = int(died.sum())
    n_default = int(df["default_within_12_months"].sum())

    ch = df["credit_history_months"]
    print(f"\ncredit_history_months: median {ch.median():.0f} "
          f"({ch.median()/12:.1f} years), range {ch.min():.0f} to {ch.max():.0f}"
          f", blank {ch.isna().sum():,}, negative {(ch < 0).sum():,}")

    print(f"\nLoans in cohort:                 {n:,}")
    print(f"Ever charged off / defaulted:    {n_died:,}  ({n_died / n:.2%})")
    print(f"  ...of which within 12 months:  {n_default:,}")
    print(f"  ...later than 12 months:       {n_died - n_default:,}")
    print(f"Borrowers who never paid at all: {int((never_paid & died).sum()):,}")

    print(f"\n>>> DEFAULT RATE (cutoff {CUTOFF_MONTHS} months, measured lag): "
          f"{n_default / n:.2%}   [{n_default:,} defaults]")
    print(f"    class balance: {n - n_default:,} non-default vs {n_default:,} default"
          f"  ({(n - n_default) / max(n_default, 1):.1f} : 1)")

    print("\n    Sensitivity to the lag assumption:")
    print(f"    {'cutoff':>8}{'implied lag':>13}{'defaults':>11}{'rate':>9}{'vs chosen':>11}")
    for c in [CUTOFF_MONTHS] + SENSITIVITY_CUTOFFS:
        k = int(label_at(c).sum())
        tag = "  <-- chosen" if c == CUTOFF_MONTHS else ""
        delta = "" if c == CUTOFF_MONTHS else f"{(k - n_default) / n * 100:+.2f} pp"
        print(f"    {c:>8}{12 - c:>13}{k:>11,}{k / n:>9.2%}{delta:>11}{tag}")

    # --- Save --------------------------------------------------------------
    out_cols = cfg.model_features() + ["default_within_12_months"]
    df[out_cols].to_parquet(OUT, index=False)
    size_mb = OUT.stat().st_size / 1024**2
    print(f"\nSaved {OUT.relative_to(ROOT)}  ({len(df):,} rows x {len(out_cols)} cols, {size_mb:.0f} MB)")
    print("Label-source columns are NOT in the saved file - only features + target.")


if __name__ == "__main__":
    main()
