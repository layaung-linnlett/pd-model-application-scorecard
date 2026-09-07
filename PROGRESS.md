# PROGRESS — pd-model-application-scorecard

> **Read this file first at the start of every session, before doing anything else.**

**Project:** Using information known at loan-application time, estimate a borrower's
probability of default within 12 months, so higher-risk applicants can be routed to
manual review instead of reviewing everyone.

**Working mode:** Claude acts as a coding tutor, not autopilot. Every section has a
done gate. Claude stops after each question and waits. No section starts until the
user explicitly says "continue".

---

## Current stage

**Stage 5 — Missing-data policy. COMPLETE 2026-09-07.**
Next: Stage 6 — logistic regression baseline. Awaiting user "continue".
OPEN ITEM to settle first: `earliest_cr_line` is a date string and must become
credit-history length at application; see Stage 5 notes below.

---

## Stage log

### Stage 0 — Repo scaffold — DONE 2026-09-05
- Created project structure: `src/`, `data/{raw,interim}/`, `outputs/{figures,models}/`,
  `notebooks/`, `tests/`, `docs/`.
- `git init` on branch `main`.
- `.gitignore` excludes `data/` (size + Lending Club redistribution terms) and `.venv/`.
- `requirements.txt` pinned, matching conventions in `customer_churn_prediction`.
  XGBoost and imbalanced-learn are deliberately commented out until the baseline is approved.
- `.venv` on Python 3.11.

### Stage 1 — Data source & target definition — IN PROGRESS

**Settled 2026-09-05:**
- Target construction = **Option B** (reconstruct the 12-month window from `last_pymnt_d`):
  `default_within_12_months = 1` if terminal status is `Charged Off`/`Default` AND
  `issue_d` -> `last_pymnt_d` gap <= 9 months (assumed 3-month lag from last payment to
  90 DPD); borrowers who never paid -> 1; `Current`/`Fully Paid`/merely `Late` -> 0.
  The 9-month cutoff is an explicit assumption and must be sensitivity-tested at 9 vs 12
  months, with the base-rate movement reported in the README.
- File will be downloaded by the user into `data/raw/`.

**Settled 2026-09-06:** dataset = `wordsforthewise/lending-club`,
file `data/raw/accepted_2007_to_2018Q4.csv` (1.6 GB, gitignored).
An earlier download of `wendykan/lending-club-loan-data` (`loan.csv`) was discarded.

Verified empirically by `src/00_inspect_raw.py` rather than trusting the filename:
- 2,260,701 rows; originations span 2007-2018
- snapshot date (max `last_credit_pull_d`) = **Apr 2019**
- 2015 cohort 421,095 + 2016 cohort 434,407 = **855,502 loans**

Last origination Dec 2016 vs Apr 2019 snapshot = >2 years of observation per loan,
comfortably more than the 12-month window plus charge-off reporting lag. No
right-censoring in this cohort.

**Stage 1 COMPLETE.**

---

## Explicitly agreed / approved

| Date | Decision |
|------|----------|
| 2026-09-05 | Project framing, target concept, and session discipline as stated by user |
| 2026-09-05 | Repo scaffold and tooling conventions |
| 2026-09-05 | Target definition: **Option B**, reconstructed 12-month window via `last_pymnt_d`, 9-month cutoff, sensitivity test required |
| 2026-09-06 | Dataset: `accepted_2007_to_2018Q4.csv`; 2015-2016 cohort of 855,502 loans; Apr 2019 snapshot verified sufficient |
| 2026-09-06 | Drop `int_rate`, `grade`, `sub_grade` (Lending Club's own risk assessment) — and `installment`, which is a function of `int_rate` |
| 2026-09-06 | Drop `zip_code` and `addr_state` on Equality Act 2010 fair-lending grounds |
| 2026-09-06 | Column policy final: 78 features / 3 label-building / 70 dropped. See `docs/leakage_checklist.md` |
| 2026-09-06 | Stay with Option B (12-month window), not Option A (lifetime) — confirmed after review |
| 2026-09-06 | Lag measured, not assumed: cutoff corrected 9 -> **8**. Default rate **3.70%**, imbalance **26.0:1** |
| 2026-09-07 | Split 60/20/20 stratified, `random_state=42`. Split done BEFORE any cleaning, to prevent preprocessing leakage |
| 2026-09-07 | Drop 14 Dec-2015 bureau columns (availability is a date stamp, not a borrower attribute). Features 78 -> **64** |
| 2026-09-07 | Missing-data policy approved: flag + train-median for numerics, "Unknown" category for `emp_length`, fills learned on train only |

## Pending (not started)

- [x] Stage 1 — Data source confirmed and verified; target definition agreed (implementation pending)
- [x] Stage 2 — Leakage checklist — COMPLETE 2026-09-06
- [x] Stage 3 — Build modelling dataset — COMPLETE 2026-09-06
- [x] Stage 4 — Stratified train/validation/test split — COMPLETE 2026-09-07
- [x] Stage 5 — Missing-data policy — COMPLETE 2026-09-07
- [ ] Stage 6 — Logistic regression baseline + coefficient walkthrough  **GATE**
      NOTE: 78 features is too many to walk through one at a time. Agree a smaller
      core set with the user before the walkthrough.
- [ ] Stage 7 — Evaluation: confusion matrix, precision, recall, ROC-AUC, KS, Gini  **GATE**
- [ ] Stage 8 — Later models (only after baseline approved; `class_weight='balanced'` before SMOTE)  **GATE**
- [ ] Stage 9 — README: business framing, leakage checklist, missing-data policy,
      Responsible-use and limitations (FCA CONC 5.2A, UK GDPR Art. 22, Equality Act 2010)


---

### Stage 2 — Leakage checklist — COMPLETE 2026-09-06

All 151 columns classified, none unallocated (verified programmatically against the
file header). Policy lives in `src/config.py`; reasoning in `docs/leakage_checklist.md`.

Teaching points the user worked through and should be able to defend in an interview:
- `last_fico_range_*` leaks (post-origination pulls) while `fico_range_*` is safe —
  the prefix, not the concept, decides it.
- `chargeoff_within_12_mths` sounds like the target but is bureau data about the
  borrower's *other* accounts at application time. Safe.
- Dropping `int_rate` is insufficient on its own: `installment` is derived from it and
  would reintroduce it. Drop functions of a dropped column too.
- `zip_code`/`addr_state` are a *fairness* exclusion, not a leakage one.


---

### Stage 3 — Modelling dataset — COMPLETE 2026-09-06

`src/01_build_dataset.py` -> `data/interim/cohort_2015_2016.parquet`
(855,502 rows x 78 features + target, 60 MB, gitignored).

**Headline numbers (FINAL, cutoff 8):**

| Measure | Value |
|---|---|
| Cohort size | 855,502 |
| Default within 12 months (the target) | **3.70%** (31,629) |
| Ever charged off / defaulted (lifetime) | 16.84% (144,056) |
| Share of all defaults occurring after month 12 | 78% |
| Class balance | 26.0 : 1 |
| Never made any payment | 811 |

### Stage 3b — the lag was MEASURED, not assumed

`src/02_measure_lag.py` tested the assumed 3-month lag against loans that were
delinquent at the Apr 2019 snapshot. Months since last payment, by status:

| Status | median months since last payment |
|---|---|
| Current (paying normally) | 1 |
| In Grace Period | 1 |
| Late (16-30 days) | 2 |
| Late (31-120 days) | 3 (p75 = 4) |

A clean ladder: each additional missed monthly payment moves the borrower one step
further behind. The 90-day point sits in the upper part of the `Late (31-120 days)`
bucket, implying a lag of **~4 months**, so the cutoff was corrected from 9 to
**12 - 4 = 8**.

**Sensitivity, after measurement:**

| Cutoff | Implied lag | Rate | Status |
|---|---|---|---|
| 8 | 4 months | **3.70%** | chosen, matches measurement |
| 9 | 3 months | 4.38% | original guess; still plausible (lag brackets 3-4) |
| 12 | 0 months | 6.51% | implausible - implies 90 DPD on the first missed payment |

**This is the headline methodological result of Stage 3.** Measuring the lag narrowed
the plausible range from 4.38-6.51% (2.13 pp of ignorance) to 3.70-4.38% (0.68 pp) -
roughly a two-thirds reduction in uncertainty. The README should say plainly:
*"I assumed a 3-month lag, measured it against the data, found it closer to 4, and
corrected the cutoff."* A residual 0.68 pp uncertainty remains and must still be
disclosed in "Responsible-use and limitations".

**Consequences for later stages:**
- Accuracy is meaningless here: predicting "never defaults" scores 96.3%. Excluded
  from the evaluation set, as agreed.
- `class_weight='balanced'` is justified by the 26.0:1 imbalance (worse than the
  21.9:1 at cutoff 9). Still to be tried BEFORE any SMOTE, per user instruction.
- User predicted 20% for the default rate, which is close to the *lifetime* rate
  (16.84%) rather than the 12-month rate. Good teaching moment: the Option A vs
  Option B distinction showing up in the data.


---

### Stage 4 — Stratified split — COMPLETE 2026-09-07

`src/03_split.py` -> `data/interim/{train,val,test}.parquet`

| Pile | Rows | Share | Defaults | Rate |
|---|---|---|---|---|
| train | 513,300 | 60.0% | 18,977 | 3.6971% |
| val | 171,101 | 20.0% | 6,326 | 3.6972% |
| test | 171,101 | 20.0% | 6,326 | 3.6972% |

Rate spread across piles: 0.0002 pp. Every row lands in exactly one pile
(asserted in the script, not just assumed).

**Design decisions:**
- Split performed BEFORE any imputation or encoding. Filling missing values using
  statistics from the whole dataset would leak test information into train.
- `random_state=42` so the split is reproducible.
- The test pile is sealed. Do not evaluate on it until the model is final.

**Teaching point the user worked through:** they predicted the test-pile rate would be
3.70/3, confusing *count* with *rate*. The split divides defaulters AND non-defaulters
together, so the count falls but the proportion holds. The output demonstrates this
directly: counts 18,977 / 6,326 / 6,326 against identical rates.

**Noted for later (not yet done):** a real credit team would usually ALSO check an
out-of-time split - train on 2015, test on 2016 - to see whether the model survives
into a later period. The user specified a random stratified split, which is what is
implemented. Worth offering as an extension before the README stage.


---

### Stage 5 — Missing-data policy — COMPLETE 2026-09-07

Full write-up in `docs/missing_data_policy.md`; lists live in `src/config.py`.

**Dropped rather than filled — 14 columns.** `src/04_missingness.py` found twelve
bureau columns missing at an identical 46.7%, on the same rows (239,558 all-missing
vs 239,571 any-missing out of 513,300). Cause: Lending Club began collecting them in
Dec 2015. Missingness by origination month is a step function - 100% through Nov 2015,
51.8% in Dec 2015, 0.0% from Jan 2016. `il_util` (13.2% missing even in 2016) and
`mths_since_rcnt_il` (2.7%) share the discontinuity and went with the block.

Their apparent predictive signal (-0.59 pp, identical across all twelve) was purely
the 2015-vs-2016 vintage difference. Keeping them, or flagging their missingness,
would have taught the model a date stamp.

Cohort was NOT reduced to 2016-only; the user chose to drop 14 columns rather than
421,095 rows.

**Policy for what remains (17 columns with blanks):**
- 12 numeric columns >= 0.5% blank: binary `<col>_missing` flag + train-median fill
- 4 numeric columns with trace blanks: median fill, no flag
- `emp_length`: blank becomes an "Unknown" category (missing 4.92% default vs
  present 3.62% - the blank is informative)
- All fill values computed on train only, applied unchanged to val/test

Features 64 -> 76 columns entering the model, before one-hot encoding.

**OPEN ITEM for Stage 6.** `earliest_cr_line` is a date string ("Aug-2003") and is
currently sitting in the feature set unusable. It should become credit-history length
in months at application = months(earliest_cr_line -> issue_d). That needs `issue_d`,
which is dropped during dataset build, so `src/01_build_dataset.py` must derive it
there and another rebuild is required. NOTE: `issue_d` itself must never become a
feature - it is the vintage, and would reproduce exactly the trap the 14 dropped
columns represent.
