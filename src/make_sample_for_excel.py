"""Utility: produce small, openable files so the data can be inspected by eye.

The raw CSV is 1.6 GB and will not open in Excel. This writes:

  outputs/sample_by_month.csv   50 loans per origination month (24 months),
                                sorted by date, with a readable subset of
                                columns - including four of the credit-bureau
                                fields that only start being collected in
                                Dec 2015, so the change is visible directly.

  outputs/figures/missingness_by_month.png
                                the same fact as a chart, next to a column
                                that is complete throughout for contrast.

Not part of the modelling pipeline. Nothing here feeds the model.
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
OUT_CSV = ROOT / "outputs" / "sample_by_month.csv"
OUT_PNG = ROOT / "outputs" / "figures" / "missingness_by_month.png"

# Columns chosen to tell the story, in a sensible reading order.
SHOW = [
    "issue_d",                      # when the loan started
    "loan_status", "last_pymnt_d",  # how it ended - used to build the label
    "loan_amnt", "term", "annual_inc", "dti", "emp_length",
    "fico_range_low", "fico_range_high",
    "home_ownership", "purpose",
    "revol_util", "open_acc",       # kept features, complete throughout
    "all_util", "open_acc_6m", "inq_last_12m", "total_bal_il",  # the Dec-2015 block
]

CLUSTER_COL = "all_util"     # only collected from Dec 2015
CONTRAST_COL = "revol_util"  # collected throughout
PER_MONTH = 50


def main() -> None:
    print("Reading cohort rows from the raw file...")
    parts = []
    for chunk in pd.read_csv(RAW, usecols=SHOW, chunksize=400_000, low_memory=False):
        issued = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
        keep = issued.between(cfg.COHORT_START, cfg.COHORT_END + "-31")
        if keep.any():
            sel = chunk.loc[keep].copy()
            sel["issue_month"] = issued.loc[keep].dt.to_period("M")
            parts.append(sel)
    df = pd.concat(parts, ignore_index=True)
    print(f"  cohort rows: {len(df):,}")

    # --- Excel-openable sample -------------------------------------------
    # Every month has far more than PER_MONTH loans, so a flat per-group
    # sample is safe. groupby.sample keeps the grouping column, unlike .apply.
    sample = (df.groupby("issue_month", observed=True)
                .sample(n=PER_MONTH, random_state=42)
                .sort_values("issue_month"))
    cols = ["issue_month"] + [c for c in SHOW if c != "issue_d"]
    sample[cols].to_csv(OUT_CSV, index=False)
    print(f"  wrote {OUT_CSV.relative_to(ROOT)}  ({len(sample):,} rows x {len(cols)} cols)")

    # --- Chart ------------------------------------------------------------
    by_month = df.groupby("issue_month").agg(
        cluster=(CLUSTER_COL, lambda s: s.isna().mean()),
        contrast=(CONTRAST_COL, lambda s: s.isna().mean()),
    )
    labels = [str(m) for m in by_month.index]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    colours = ["#c0392b" if v > 0.4 else "#95a5a6" for v in by_month["cluster"]]
    ax1.bar(labels, by_month["cluster"] * 100, color=colours)
    ax1.set_ylabel("% missing")
    ax1.set_title(f"'{CLUSTER_COL}' — a field Lending Club only starts collecting in Dec 2015",
                  fontsize=12, loc="left", weight="bold")
    ax1.set_ylim(0, 105)
    ax1.annotate("collected from here on", xy=(11.6, 52), xytext=(13.5, 72),
                 arrowprops=dict(arrowstyle="->", color="#2c3e50"), fontsize=10)

    ax2.bar(labels, by_month["contrast"] * 100, color="#27ae60")
    ax2.set_ylabel("% missing")
    ax2.set_title(f"'{CONTRAST_COL}' — a field collected throughout, for contrast",
                  fontsize=12, loc="left", weight="bold")
    ax2.set_ylim(0, 105)

    plt.xticks(rotation=90, fontsize=8)
    fig.suptitle("Why 14 columns had to be dropped: they are a date stamp, not a borrower attribute",
                 fontsize=13, weight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"  wrote {OUT_PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
