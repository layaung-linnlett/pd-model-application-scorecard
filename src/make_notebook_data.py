"""Build a small companion file so the teaching notebook runs in seconds.

The notebook needs a handful of raw columns that the modelling dataset
deliberately throws away (loan_status, last_pymnt_d, issue_d) plus two used to
demonstrate the Dec-2015 missingness cliff. Reading the 1.6 GB file for those
every time would make the notebook unusable, so they are extracted once here.

This file is for teaching only. Nothing in the modelling pipeline reads it.
"""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
OUT = ROOT / "data" / "interim" / "notebook_demo.parquet"

COLS = ["issue_d", "loan_status", "last_pymnt_d", "last_credit_pull_d",
        "earliest_cr_line", "all_util", "revol_util", "dti", "annual_inc"]


def main() -> None:
    parts = []
    for chunk in pd.read_csv(RAW, usecols=COLS, chunksize=400_000, low_memory=False):
        issued = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
        keep = issued.between(cfg.COHORT_START, cfg.COHORT_END + "-31")
        if keep.any():
            parts.append(chunk.loc[keep])
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(OUT, index=False)
    print(f"{len(df):,} rows x {df.shape[1]} cols -> {OUT.relative_to(ROOT)} "
          f"({OUT.stat().st_size / 1024**2:.0f} MB)")


if __name__ == "__main__":
    main()
