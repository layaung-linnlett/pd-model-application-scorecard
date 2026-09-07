# Data-quality decision log

Every extreme value that was flagged, what the evidence said, what was decided, and why.

**The principle:** a statistical outlier is not automatically bad data. In credit risk,
an extreme value is often the borrower the model most needs to catch. Values were
therefore only treated where the evidence showed they were **impossible**, not merely rare.

**The treatment, where one was applied:** set the value to **missing** — not deleted,
not capped.

- **Not deleted**, because one bad field should not discard a borrower's other 60 valid columns.
- **Not capped**, because capping asserts a value we do not believe. Missing says
  "we do not know", which is true, and hands the row to the policy already agreed in
  [`missing_data_policy.md`](missing_data_policy.md).

All bounds are in `DATA_QUALITY_LIMITS` in [`../src/config.py`](../src/config.py).
All figures below are from the 2015–2016 cohort (855,502 loans, 3.70% default rate).

---

## Corrected — evidence says impossible

| Feature | Evidence | Decision | Business reason |
|---|---|---|---|
| `annual_inc` < $5,000 | 99 borrowers. **93 of 99 borrowed more than their entire stated annual income.** 72 of 99 were never income-verified by Lending Club. 59 state exactly $0. Their median `dti` is 385. | Set to missing (99 values) | No lender advances $15,000 against a verified $0 income. The figure is a form-entry failure, and it is the root cause of the `dti` problem below. |
| `dti` outside 0–100 | 136 values. Default risk **rises** with dti to ~39 (6.41% vs 3.70% base) then **collapses** above 100 (1.23%). 5 rows are exactly 999. | Set to missing (136 values) | Real financial stress does not reverse. `dti` = debt ÷ income, so the near-zero incomes above produce absurd ratios. Same defect, second column. |
| `revol_util` > 150% | 14 values. | Set to missing | Being over your credit limit is real and common. Being 50% over your total limit across all cards is not credible. |
| `total_rev_hi_lim` = 9,999,999 | 1 value. | Set to missing | An all-9s placeholder, not a credit limit. |
| `pub_rec` > 20 | 23 borrowers, values up to 86. Correlates 0.72 with `tax_liens`; 22 of the 23 have both above 20. | Set to missing (23 values) | 61 bankruptcies and judgments against one person is not credible. **Safe to clear because the dose-response is flat** — see below. |
| `tax_liens` > 20 | 22 borrowers, same rows as above. | Set to missing (22 values) | Same defect, same rows. |

**Total: 295 values cleared out of 855,502 rows — 0.03% of the data.**

### Why clearing `pub_rec` loses nothing

| `pub_rec` | Borrowers | Default rate |
|---|---|---|
| 0 | 698,174 | 3.59% |
| 1 | 126,749 | **4.22%** |
| 2–3 | 25,962 | 3.97% |
| 4–7 | 4,182 | 3.59% |
| 8–20 | 412 | 3.64% |
| 21+ | 23 | 4.35% |

The only real step is 0 → 1. Beyond the first record the count carries no signal, so
clearing an implausible tail cannot destroy information that was never there.

---

## Kept — rare, but genuinely informative

| Feature | Evidence | Decision | Business reason |
|---|---|---|---|
| `dti` 39–100 | 5,022 borrowers, **6.41% default** vs 3.70% base | **Keep** | The single strongest risk marker found. Capping at the 99th percentile would have flattened it — exactly the borrowers manual review exists to catch. |
| `revol_util` 100–150% | 2,331 borrowers, **4.66% default** | **Keep** | Being over your credit limit is real, common, and predictive of trouble. |
| `delinq_amnt` > 0 | 2,510 borrowers, **5.06% default** | **Keep** | Money currently overdue. Rare because most borrowers owe nothing, not because it is erroneous. |
| `tot_coll_amt` high | 5,132 borrowers, 3.02% default | **Keep** | Directional signal. |
| `annual_inc` > $1m | 119 borrowers, 3.36% default | **Keep** | Wealthy borrowers exist. Capping at the 99th percentile ($270,000) would erase genuine high earners. |
| `total_rev_hi_lim` > $165k | 5,127 borrowers, 2.28% default | **Keep** | Large credit limits are protective and real. |

---

## Resolved by other means

| Feature | Evidence | Decision | Business reason |
|---|---|---|---|
| `credit_history_months` 600–840 | 949 borrowers, 6.8% default. Investigated: `earliest_cr_line` dates smoothly distributed 1946–1965 with no clustering; income, dti, loan size and utilisation identical to everyone else; better Lending Club grades; 47.3% with 10+ years employment vs 36.3%. **Genuine older borrowers, not date errors.** | Feature removed entirely at Stage 6c | Not removed for data quality — removed because it acts as an age proxy and a pre-registered test showed it added nothing. See `PROGRESS.md` Stage 6c. |

---

## Effect on the model

| | Before | After |
|---|---|---|
| Highest probability assigned to any borrower | 100.0% | **56.3%** |
| Borrowers flagged at a 0.5 threshold | 14 | 2 |
| ROC-AUC (validation) | 0.6956 | 0.6976 |
| Defaulters caught at 5% review capacity | 966 | **971** |

The model can no longer be near-certain about a borrower on the strength of a
form-entry error. One `dti` value of 999 sat 101 standard deviations from the mean and
contributed roughly 100× what a normal borrower's `dti` contributed, drowning the other
99 columns.

## What was deliberately NOT done

**No percentile capping (winsorising) was applied.** It was considered and rejected on
evidence: at the 1st/99th percentile it would have altered roughly 10,000 rows per
column, against 295 values in total under the impossible-value rules — and it would
have flattened every one of the genuine signals in the "Kept" table above.

## Known limitation

Large outliers remain in `tot_coll_amt` (344 SDs), `delinq_amnt` (168 SDs),
`annual_inc` (131 SDs) and `revol_bal` (119 SDs). These are retained deliberately —
each carries directional signal — but they do give a small number of borrowers
substantial influence on a linear model. A production scorecard would normally address
this by binning variables into bands (weight-of-evidence transformation) rather than
by capping. That was out of scope here and is noted as future work.
