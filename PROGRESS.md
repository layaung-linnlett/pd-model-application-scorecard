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

**Stage 3 + 3b — Modelling dataset built, lag assumption measured and corrected.
COMPLETE 2026-09-06.**
Next: Stage 4 — stratified train/validation/test split. Awaiting user "continue".

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

## Pending (not started)

- [x] Stage 1 — Data source confirmed and verified; target definition agreed (implementation pending)
- [x] Stage 2 — Leakage checklist — COMPLETE 2026-09-06
- [x] Stage 3 — Build modelling dataset — COMPLETE 2026-09-06
- [ ] Stage 4 — Stratified train/validation/test split  **GATE**
- [ ] Stage 5 — Logistic regression baseline + coefficient walkthrough  **GATE**
      NOTE: 78 features is too many to walk through one at a time. Agree a smaller
      core set with the user before the walkthrough.
- [ ] Stage 6 — Evaluation: confusion matrix, precision, recall, ROC-AUC, KS, Gini  **GATE**
- [ ] Stage 7 — Later models (only after baseline approved; `class_weight='balanced'` before SMOTE)  **GATE**
- [ ] Stage 8 — README: business framing, leakage checklist, missing-data policy,
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
