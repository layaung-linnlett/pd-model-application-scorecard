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

**Stage 6d — Data-quality decision log COMPLETE. Outlier question CLOSED.**
Next: `class_weight='balanced'` (must be tried before any SMOTE), then Stage 7
evaluation. Baseline itself is still awaiting explicit user approval.

`notebooks/01_walkthrough.ipynb` reproduces every stage so far with live output.
Regenerate it with:
    ./.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/01_walkthrough.ipynb

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

**RESOLVED 2026-09-07 — `earliest_cr_line` converted.** It was a date string
("Aug-2003") that a model cannot do arithmetic on. `src/01_build_dataset.py` now
derives `credit_history_months = months(earliest_cr_line -> issue_d)` and drops the
raw date. Feature count unchanged at 64 (one swapped for one). `cfg.model_features()`
is the accessor for the model's column list; `cfg.FEATURES` still contains
`earliest_cr_line` so the 151-column classification stays complete.

`issue_d` is used for the subtraction and immediately discarded. It must NEVER be a
feature: it is the loan vintage, and would reproduce exactly the trap that
DROP_TIME_VARYING_AVAILABILITY exists to prevent.

Sanity checks on the derived column: median 182 months (15.2 years), min 37, max 999,
**0 blanks, 0 negatives**. The 999 was investigated and is genuine - the borrower's
first credit account dates to Mar-1933 - not a sentinel value. The distribution is
smooth (99% at 486, 99.9% at 603); 4 loans exceed 70 years and are probably data-entry
errors, but at 4 in 855,502 they cannot shift the fit.

DECISION: no capping/winsorising before the baseline. Revisit only if the baseline
shows sensitivity to extreme values. Record as a known limitation in the README.


---

### Stage 6 — Baseline logistic regression — FITTED, NOT APPROVED 2026-09-07

`src/05_baseline_logistic.py` -> `outputs/models/baseline_logistic.joblib`,
`outputs/baseline_coefficients.csv`

Plain LogisticRegression, no class weighting, no resampling - the honest starting
point, per the user's instruction that `class_weight='balanced'` comes later and
SMOTE later still.

Pipeline: 12 missing-flags added outside the pipeline (deterministic per row, so no
leakage), then median fill + StandardScaler on 58 numerics, passthrough on flags,
"Unknown" fill + one-hot on 6 text columns. 100 columns after encoding. Converged in
41 iterations.

| Measure | Value |
|---|---|
| ROC-AUC (validation) | 0.6969 |
| Flagged at prob >= 0.5 | 15 of 171,101 |
| Actually defaulted | 6,326 |

The 15-vs-6,326 result is the imbalance problem made concrete and is expected, not a
bug: at a 3.70% base rate the model is rarely more than 50% confident about anyone.
Motivates `class_weight='balanced'` next.

**BLOCKER - outliers. Decision pending.** One borrower scored 99.99994%. Diagnosis:
`dti = 999` (a placeholder; debt payments at ten times income), sitting 101 standard
deviations out. Because logistic regression multiplies and adds, that single value
swamped the other 99 columns.

This is widespread, not isolated:

| column | worst value | SDs from mean |
|---|---|---|
| tot_coll_amt | 848,438 | 344 |
| total_rev_hi_lim | 9,999,999 (all-9s placeholder) | 266 |
| annual_inc | 9,573,072 | 131 |
| revol_bal | 2,904,836 | 119 |
| dti | 999 | 101 |

Only 81 of 513,300 training rows have `dti >= 100`, and 560 cells out of 17.1m exceed
20 SDs - a tiny tail with outsized influence.

PROPOSED (awaiting user approval): winsorise every numeric feature at its 0.1st and
99.9th percentile, learned on train only, inside the pipeline. 99.9 rather than 99 so
genuinely high earners are preserved. Then re-fit and run the coefficient walkthrough.

NOTE: this reverses the Stage 5 decision to defer capping. That decision was explicitly
conditional - "revisit only if the baseline shows sensitivity to extreme values" - and
the baseline showed exactly that.

**Also flagged for the walkthrough:** `OneHotEncoder(drop="first")` dropped a rare
`home_ownership` category as the reference level, so all three visible home_ownership
coefficients are negative and hard to read. Consider dropping the most COMMON category
instead, or not dropping at all, before interpreting coefficients.


---

### Stage 6b — PRE-REGISTERED DECISION RULE, set 2026-09-07 BEFORE the test was run

**Question:** should `credit_history_months` stay in the model?

**Why it is in question.** Its coefficient is +0.056, third weakest of the core ten, yet
it may act as a proxy for AGE across its full range (longer histories require more time
to accumulate, so the value places a floor under the borrower's age). Age is a protected
characteristic under the Equality Act 2010. Known fairness cost, weak measured benefit.

**Burden of proof sits on RETAINING the feature**, not on dropping it. A potentially
sensitive proxy has to earn its place; it is not entitled to one just by being available.

**The criterion — chosen by the user, recorded before any result was seen:**

> Retain `credit_history_months` only if it delivers at least a **1% relative
> improvement in defaulters caught at 5% review capacity** on the validation pile.
> Otherwise drop it.

Operationally: rank the 171,101 validation borrowers by predicted probability, take the
riskiest 5% (8,555 borrowers - the number a lender could realistically review by hand),
and count how many of the 6,326 actual defaulters fall inside that set, with and without
the feature.

**ROC-AUC is explicitly NOT the criterion.** It averages over thresholds the business
would never operate at. The decision is about performance in the review region only.

This rule is committed before the experiment so the threshold cannot be adjusted to fit
whatever result appears. Result to follow in Stage 6c.


---

### Stage 6c — Age-proxy features removed — DONE 2026-09-07

**Result of the pre-registered test** (validation, riskiest 5% = 8,555 reviewed,
6,326 actual defaulters):

| model | defaulters caught | recall | ROC-AUC |
|---|---|---|---|
| keep everything | 964 | 15.24% | 0.6969 |
| drop `credit_history_months` only | 972 | 15.37% | 0.6969 |
| drop the 0.92 pair | 960 | 15.18% | 0.6961 |
| drop all 8 "age-linked" | 961 | 15.19% | 0.6948 |

All four sit inside the +/-31 expected from chance. **There is no measurable
difference between them.** Performance therefore had no vote in the decision.

**Why dropping only `credit_history_months` would have been cosmetic.**
`mo_sin_old_rev_tl_op` correlates with it at **0.92** - it is the same measurement
("how long ago did you start using credit") under another name. Removing one while
keeping the other changes the feature list and not the model. That is why the
single-feature test showed no loss: the information never left.

**Correction made during the analysis.** An initial grouping of 8 "age-linked"
features was too broad. Only three are direct time-since-first-credit measures:

    earliest_cr_line -> credit_history_months   (the derived feature, now gone)
    mo_sin_old_rev_tl_op                        corr 0.92
    mo_sin_old_il_acct                          corr 0.35

`mo_sin_rcnt_rev_tl_op` and `mo_sin_rcnt_tl` measure how RECENTLY an account was
opened - current behaviour, not accumulated time - and were wrongly included.
`mort_acc`, `total_acc`, `num_rev_accts` correlate only 0.26-0.31; having four credit
cards is not a measure of age. All five retained.

**ACTION TAKEN:** dropped `earliest_cr_line`, `mo_sin_old_rev_tl_op`,
`mo_sin_old_il_acct`. Features 64 -> **61**. Missing-flags 12 -> 11
(`mo_sin_old_il_acct` had one). The `credit_history_months` derivation is removed
from `src/01_build_dataset.py`; `DATE_FEATURES_TO_DERIVE` and `DERIVED_FEATURES` are
now empty but retained as hooks for the inventory tooling.

Rebuilt, re-split, re-fitted. Cohort, default rate and split are unchanged
(855,502 / 3.70% / 26:1). Re-fit: 96 encoded columns, ROC-AUC 0.6956,
**966 defaulters caught at 5% review** (vs 964 before - noise).

**IMPORTANT for the README - do not overclaim.** This does NOT make the model
age-blind. Account-count features still correlate with age at 0.26-0.31 and are
retained. The honest claim is: *"direct age proxies were identified and removed at
no measured cost; weaker residual correlation remains and is disclosed."*

**Method worth writing up.** The decision rule was committed to git (commit 5c060df)
BEFORE the experiment was run, so the threshold could not be adjusted to fit the
result. That is the defensible way to make this kind of call.


---

### Stage 6d — Data-quality decision log — COMPLETE 2026-09-07

Full log in `docs/data_quality_decision_log.md`. Bounds in `DATA_QUALITY_LIMITS`
(`src/config.py`), applied by `apply_data_quality_limits()` in `01_build_dataset.py`.

**Percentile capping was CONSIDERED AND REJECTED.** At 1st/99th it would have altered
~10,000 rows per column and flattened genuine risk signals. Rejected on the user's own
evidence, not on preference.

**Treatment chosen: impossible values -> MISSING.** Not deleted (one bad field should
not discard 60 good columns), not capped (capping asserts a value we do not believe).
Missing is honest and reuses the agreed missing-data policy.

**295 values cleared out of 855,502 rows (0.03%):**
annual_inc < $5,000 (99), dti outside 0-100 (136), revol_util > 150 (14),
total_rev_hi_lim = 9,999,999 (1), pub_rec > 20 (23), tax_liens > 20 (22).

**Two findings that shaped the decisions:**
1. Low income and dti=999 are ONE defect, not two. 93 of the 99 sub-$5,000-income
   borrowers borrowed more than their stated annual income and 72 were unverified;
   dti = debt/income, so the broken income field produces the absurd ratio.
2. `pub_rec` dose-response is flat beyond the first record (0 -> 3.59%, 1 -> 4.22%,
   then 3.97 / 3.59 / 3.64%). The signal is "has any public record"; the count carries
   nothing. Clearing the implausible tail therefore cannot lose information.

**Kept deliberately, with evidence:** dti 39-100 (6.41% default), revol_util 100-150
(4.66%), delinq_amnt > 0 (5.06%), annual_inc > $1m, high tot_coll_amt, high
total_rev_hi_lim. These are the borrowers manual review exists to find.

**Effect:** highest assigned probability 100.0% -> 56.3%; flagged at 0.5 threshold
14 -> 2; ROC-AUC 0.6956 -> 0.6976; defaulters caught at 5% review 966 -> 971.

**Residual limitation for the README:** tot_coll_amt (344 SDs), delinq_amnt (168),
annual_inc (131), revol_bal (119) still carry large leverage. Retained because each
carries directional signal. A production scorecard would bin variables
(weight-of-evidence) rather than cap. Noted as future work, not done here.
