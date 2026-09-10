"""One place to see every column and what happened to it.

Run this any time you want the full picture:

    ./.venv/bin/python src/column_inventory.py

Prints a summary to the screen and writes outputs/column_inventory.csv,
which opens in Excel: one row per original column, with its status, the
reason for that status, how much of it is missing, and what we do about it.

Nothing here changes any data. It only reports.
"""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg
from glossary import MEANING

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
OUT = ROOT / "outputs" / "column_inventory.csv"

# group name -> (status, plain-English reason)
GROUPS = {
    "LABEL_SOURCE": ("USED FOR TARGET",
                     "Used to build the target. Never given to the model."),
    "APPLICATION_STATED": ("KEEP",
                           "Stated by the borrower on the application form."),
    "BUREAU_CORE": ("KEEP",
                    "Credit-bureau fact known when they applied."),
    "BUREAU_EXTENDED": ("KEEP",
                        "Credit-bureau fact known when they applied."),
    "DROP_LEAKAGE": ("DROP",
                     "LEAKAGE: recorded after the loan was granted, so it "
                     "does not exist when a real applicant applies."),
    "DROP_LENDER_JUDGEMENT": ("DROP",
                              "Lending Club's own risk score (or derived from "
                              "it). Using it means copying their homework."),
    "DROP_TIME_VARYING_AVAILABILITY": ("DROP",
                                       "Only collected from Dec 2015. Tells you "
                                       "WHEN the loan was issued, not who the "
                                       "borrower is."),
    "DROP_AGE_PROXY": ("DROP",
                       "Measures how long ago they started using credit, which "
                       "stands in for age. Removed after testing showed it "
                       "costs nothing."),
    "DROP_LENDER_PROCESS": ("DROP",
                            "Records Lending Club's own triage process, not the "
                            "borrower. Would mean something different at another "
                            "lender."),
    "DROP_FAIRNESS_PROXY": ("DROP",
                            "Location. Stands in for protected characteristics "
                            "(Equality Act 2010)."),
    "DROP_JOINT_SPARSE": ("DROP",
                          "Only filled in for joint applications, which are "
                          "rare here. Almost always empty."),
    "DROP_NON_PREDICTIVE": ("DROP",
                            "Identifier, free text, constant, or a duplicate "
                            "of a column we already keep."),
}


def treatment_for(col: str, pct_missing: float, status: str) -> str:
    if status == "DROP":
        return "-"
    if status == "USED FOR TARGET":
        return "builds the target, then discarded"
    if col in cfg.DATE_FEATURES_TO_DERIVE:
        return "TO DO: convert date to credit-history length in months"
    if col == "emp_length":
        return "blank becomes an 'Unknown' category"
    if col in cfg.CATEGORICAL_FEATURES:
        return "one-hot encode (text -> columns of 0/1)"
    if col in cfg.MISSING_FLAG_COLUMNS:
        return "add a '_missing' flag, then fill with the training median"
    if col in cfg.MEDIAN_FILL_ONLY:
        return "fill with the training median (too few blanks to flag)"
    if pct_missing > 0:
        return "fill with the training median"
    return "use as-is (nothing missing)"


def main() -> None:
    header = pd.read_csv(RAW, nrows=0).columns.tolist()

    lookup = {}
    for group, (status, reason) in GROUPS.items():
        for col in getattr(cfg, group):
            lookup[col] = (group, status, reason)

    print(f"Counting blanks across the {cfg.COHORT_START}..{cfg.COHORT_END} "
          f"cohort (this reads the 1.6 GB file, ~2 min)...")
    nulls = pd.Series(0, index=header, dtype="int64")
    rows = 0
    for chunk in pd.read_csv(RAW, chunksize=200_000, low_memory=False):
        issued = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
        keep = issued.between(cfg.COHORT_START, cfg.COHORT_END + "-31")
        if keep.any():
            sel = chunk.loc[keep]
            rows += len(sel)
            nulls += sel.isna().sum()
    pct = (nulls / rows * 100).round(2)

    records = []
    for col in header:
        group, status, reason = lookup[col]
        records.append({
            "column": col,
            "meaning": MEANING[col],
            "status": status,
            "why": reason,
            "pct_blank": pct[col],
            "what_we_do": treatment_for(col, pct[col], status),
            "group": group,
        })
    inv = pd.DataFrame(records)

    order = {"KEEP": 0, "USED FOR TARGET": 1, "DROP": 2}
    inv = inv.sort_values(["status", "pct_blank"],
                          key=lambda s: s.map(order) if s.name == "status" else -s)
    inv.to_csv(OUT, index=False)

    # ---- screen summary --------------------------------------------------
    print(f"\nCohort rows counted: {rows:,}\n")
    print("=" * 74)
    print(f"{'STATUS':<18}{'COLUMNS':>9}   WHY")
    print("=" * 74)
    for group, (status, reason) in GROUPS.items():
        n = len(getattr(cfg, group))
        short = reason.split(".")[0]
        print(f"{status:<18}{n:>9}   {short}")
    print("-" * 74)
    print(f"{'TOTAL':<18}{len(header):>9}")

    keep = inv[inv.status == "KEEP"]
    blanks = keep[keep.pct_blank > 0].sort_values("pct_blank", ascending=False)
    print(f"\n{len(keep)} columns kept as features. "
          f"{len(blanks)} of them have blanks:\n")
    print(f"  {'column':<34}{'blank':>8}   what we do")
    print("  " + "-" * 70)
    for _, r in blanks.iterrows():
        print(f"  {r.column:<34}{r.pct_blank:>7.1f}%   {r.what_we_do}")

    print(f"\n  {len(keep) - len(blanks)} kept columns have no blanks at all.")
    print(f"\nWritten to {OUT.relative_to(ROOT)} - open this in Excel.")
    print("Column B is a plain-English description of what each field means.")


if __name__ == "__main__":
    main()
