"""Stage 1 check: confirm the raw file actually covers the cohort we need.

Reads only the date columns, in chunks, so this never loads 1.6 GB into memory.
Answers three questions:
  1. What range of origination dates (issue_d) does the file contain?
  2. When was the data snapshot taken (max last_credit_pull_d)?
  3. How many loans fall in our 2015-2016 cohort?
"""

from pathlib import Path
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
COLS = ["issue_d", "last_credit_pull_d"]

issue_counts = {}
max_pull = None
total = 0

for chunk in pd.read_csv(RAW, usecols=COLS, chunksize=250_000, low_memory=False):
    total += len(chunk)

    issued = pd.to_datetime(chunk["issue_d"], format="%b-%Y", errors="coerce")
    for period, n in issued.dt.to_period("Y").value_counts().items():
        issue_counts[period] = issue_counts.get(period, 0) + n

    pulled = pd.to_datetime(chunk["last_credit_pull_d"], format="%b-%Y", errors="coerce").max()
    if pd.notna(pulled) and (max_pull is None or pulled > max_pull):
        max_pull = pulled

years = pd.Series(issue_counts).sort_index()

print(f"\nTotal rows in file: {total:,}")
print(f"Earliest origination: {years.index.min()}")
print(f"Latest origination:   {years.index.max()}")
print(f"Snapshot date (latest credit pull): {max_pull:%b %Y}")

print("\nLoans per origination year:")
for year, n in years.items():
    mark = "  <-- our cohort" if str(year) in ("2015", "2016") else ""
    print(f"  {year}  {n:>9,}{mark}")

cohort = years.get(pd.Period("2015"), 0) + years.get(pd.Period("2016"), 0)
print(f"\n2015-2016 cohort size: {cohort:,} loans")
