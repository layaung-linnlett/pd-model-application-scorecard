# PROGRESS — pd-model-application-scorecard

> **Read this file first at the start of every session, before doing anything else.**

---

## ⏸  RESUME HERE  (updated 2026-09-09)

Everything is committed and the working tree is clean.

**Stage 8 — evaluation COMPLETE 2026-09-09.** All six metrics covered with the user
interpreting each before explanation. `src/07_evaluate.py` +
`outputs/figures/baseline_evaluation.png`.

**Stage 10 COMPLETE 2026-09-10 — segment analysis done, `verification_status` dropped.**
Model is now 60 features, Gini 0.3923, recall 26.29% at 10% review capacity.

**Next: tree models (Random Forest / XGBoost), then the README.**

Bar structure agreed for the trees but NOT yet pre-registered - do that before running:
- under +50 defaulters: reject, inside the noise
- +50 to +250: real but modest. Keep logistic regression, report the tree as a
  challenger showing what accuracy costs in explainability
- over +250 (~15%): large enough to argue for giving up per-applicant explanations
The middle band mirrors real practice - lenders often build a GBM challenger and
still deploy the scorecard, because UK GDPR Art. 22 explainability outranks a few
points of Gini.

**Test pile is still sealed** and must stay so until model selection is finished.

SMOTE is still untested. Expectation after the class_weight result is that it will not
help ranking either, but that should be tested with a pre-registered bar, not assumed.

**Working style reminders — the user asked for these explicitly:**
- Cassie Kozyrkov-style: ask a question they can answer from ordinary life BEFORE
  explaining. Their answer is usually the technical policy restated.
- One idea per message. Short. Define every term. No stacked tables while teaching.
- Do not write project code without asking; they want to attempt it.
- If they say they are lost, stop and find the last solid ground - do not patch the
  current step.
- Show real output. They cannot open the 1.6 GB CSV; use
  `notebooks/01_walkthrough.ipynb`, `outputs/column_inventory.csv`, or a small sample.

**Project:** Using information known at loan-application time, estimate a borrower's
probability of default within 12 months, so higher-risk applicants can be routed to
manual review instead of reviewing everyone.

**Working mode:** Claude acts as a coding tutor, not autopilot. Every section has a
done gate. Claude stops after each question and waits. No section starts until the
user explicitly says "continue".

---

## Current stage

**Stage 7 COMPLETE 2026-09-09 — `class_weight='balanced'` tested and REJECTED.**
Baseline stands unchanged. Next: Stage 8 evaluation.

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
| 2026-09-07 | Drop 3 direct age proxies after a pre-registered test (commit 5c060df). Features 64 -> 61 |
| 2026-09-07 | Reject percentile capping; clear 295 impossible values to missing instead |
| 2026-09-07 | **BASELINE LOGISTIC REGRESSION APPROVED.** 61 features, ROC-AUC 0.6976, 971 of 6,326 defaulters caught at 5% review capacity |
| 2026-09-09 | `class_weight='balanced'` tested and **rejected** — same ranking, no gain. Keep the simpler model |
| 2026-09-09 | Operating point set to **10% review capacity** (`cfg.REVIEW_CAPACITY`), stated as an assumption, with the full curve reported alongside |

## Pending (not started)

- [x] Stage 1 — Data source confirmed and verified; target definition agreed (implementation pending)
- [x] Stage 2 — Leakage checklist — COMPLETE 2026-09-06
- [x] Stage 3 — Build modelling dataset — COMPLETE 2026-09-06
- [x] Stage 4 — Stratified train/validation/test split — COMPLETE 2026-09-07
- [x] Stage 5 — Missing-data policy — COMPLETE 2026-09-07
- [ ] Stage 6 — Logistic regression baseline + coefficient walkthrough  **GATE**
      NOTE: 78 features is too many to walk through one at a time. Agree a smaller
      core set with the user before the walkthrough.
- [x] Stage 6 — Logistic regression baseline — **APPROVED 2026-09-07**
- [x] Stage 7 — `class_weight='balanced'` tested, rejected — COMPLETE 2026-09-09
- [ ] Stage 8 — Evaluation: confusion matrix, precision, recall, ROC-AUC, KS, Gini  **GATE**
- [ ] Stage 9 — Later models (SMOTE only if class_weight proves insufficient; then trees)  **GATE**
- [ ] Stage 10 — README: business framing, leakage checklist, missing-data policy,
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


---

### Stage 7 — class_weight='balanced' — TESTED AND REJECTED 2026-09-09

`src/06_class_weight.py`. Only one thing differs between the two models:
`class_weight="balanced"` on the LogisticRegression. Same features, same split, same
preprocessing.

**What changed — a lot:**

| | flagged at 0.5 | mean predicted probability |
|---|---|---|
| baseline | 2 | 3.69% |
| balanced | 64,304 | 44.64% |

**What did not change — anything that matters:**

| | defaulters caught in riskiest 5% | recall | ROC-AUC |
|---|---|---|---|
| baseline | 971 | 15.35% | 0.6976 |
| balanced | 966 | 15.27% | 0.6982 |

Difference -5, well inside the +/-31 expected from chance.

**Why:** it is the same ranking. Of the 8,555 borrowers each model sends to review,
**7,817 are the same people (91.4% overlap)**, and the Spearman rank correlation
between the two sets of scores is **0.9934**. Class weighting changes how worried the
model is, not who it worries about. For a linear model it essentially shifts the
intercept, which moves every probability without reordering anyone.

**DECISION: reject. Keep the simpler baseline.** When two options perform the same,
take the simpler one - it is easier to explain and has fewer moving parts. Complexity
has to earn its place.

**Wider lesson worth putting in the README.** The "only 2 borrowers flagged" result was
never a failure. It was an artefact of judging the model at a 0.5 cut-off that nobody
would use. The business rule is "review the riskiest 5%", which depends only on the
ORDER of the scores, never on their absolute size. If more people need flagging, move
the threshold - do not reweight the model.

**Implication for SMOTE (still untested).** SMOTE is a more invasive form of the same
rebalancing idea. Since weighting moved the ranking by 0.7%, the prior is that SMOTE
will not help either. Test it with a pre-registered bar rather than assuming - the same
discipline used for `credit_history_months` at commit 5c060df.


---

### Stage 8 — Evaluation — IN PROGRESS from 2026-09-09

**Operating point decided: 10% review capacity.** Earlier numbers in this file used 5%,
which was Claude's arbitrary pick and was never justified. 10% is equally an assumption,
but it is now stated as one in `cfg.REVIEW_CAPACITY` and the full curve is reported
beside it, so a reader sees the trade-off rather than a number pulled from nowhere.

| Review capacity | People reviewed | Defaulters caught | Recall | vs random |
|---|---|---|---|---|
| 1% | 1,711 | 241 | 3.8% | 3.81x |
| 5% | 8,555 | 971 | 15.3% | 3.07x |
| **10%** | **17,110** | **1,678** | **26.5%** | **2.65x** |
| 20% | 34,220 | 2,778 | 43.9% | 2.20x |
| 50% | 85,550 | 4,814 | 76.1% | 1.52x |
| 100% | 171,101 | 6,326 | 100.0% | 1.00x |

**Two points from this table for the README:**
- The model's edge is largest at the top (3.8x at 1%, falling to 1.5x at 50%). Ranking
  is worth most exactly when capacity is tightest.
- At 10% capacity, **73.5% of defaulters are still approved without review**. State this
  plainly. The model reduces losses; it does not prevent them.

**All metrics COMPLETE 2026-09-09.** `src/07_evaluate.py`, figure at
`outputs/figures/baseline_evaluation.png`. Evaluated on VALIDATION; test stays sealed.

| Metric | Value | Plain meaning |
|---|---|---|
| Recall @10% | 26.53% | of 6,326 defaulters, 1,678 caught |
| Precision @10% | 9.81% | 1 useful file in 10, vs 1 in 27 at random |
| ROC-AUC | 0.6976 | pick a defaulter and a non-defaulter; 70% of the time the defaulter scores higher |
| Gini | 0.3952 | = 2*AUC-1, rescaled so useless = 0. The industry convention |
| KS | 29.0 | widest separation, occurring 34.4% down the list |
| Lift @10% | 2.65x | vs random selection |

Confusion matrix at 10%: TP 1,678 / FP 15,432 / FN 4,648 / TN 149,343.

**Accuracy is deliberately excluded.** "Never defaults" scores 96.3% and is worthless.

**Points the user reasoned out and should be able to defend:**
- A missed defaulter (~£7,000) costs ~175x a wasted review (~£40). That asymmetry is
  why low precision is acceptable here and would not be in, say, medicine.
- Recall and precision come from the SAME cell (1,678), divided by the column vs the row.
- AUC's floor is 0.50, not 0 - a random-scoring model wins half its pairwise
  comparisons. Below 0.50 means systematically wrong, which is usable upside down.
- 0.70 AUC is near the ceiling for application-only data; ~0.85 on this dataset would
  imply leakage.
- **KS peaks at 34.4% but the operating point is 10%.** KS describes the model, not the
  staffing. Setting capacity from where a statistic peaks would let a metric make a
  business decision. The user asked exactly this ("so should i use 40%?") and the
  distinction was drawn explicitly.

**Honest limitation for the README:** at 10% capacity, 4,648 defaulters (73.5%) are
approved with no review. The model reduces losses; it does not prevent them. The
score-distribution panel of the figure shows why - the two groups overlap heavily.

**Teaching note.** The user got lost when three ideas were stacked in one message
(capacity is a choice / edge is biggest at the top / most defaulters still slip through).
Splitting them into three short messages with a check after each worked. Keep doing that.


---

### Stage 9 — PRE-REGISTERED BAR FOR SMOTE, set 2026-09-09 BEFORE the test was run

**Question:** does SMOTE earn a place in the pipeline?

**What SMOTE does.** The training data has 26 non-defaulters per defaulter. SMOTE
invents synthetic defaulters - taking a real defaulter, finding a similar one, and
creating a fake borrower between them - until the classes are balanced. The model then
trains partly on borrowers who never applied for a loan.

**The criterion, fixed before any result was seen:**

> Keep SMOTE only if it catches at least **1,728 defaulters** at the 10% operating
> point on validation, i.e. **+50 (about +3%)** on the baseline's 1,678.

**Why +50 and not +17 (the 1% bar used for `credit_history_months`).** A count of ~1,678
carries roughly sqrt(1678) = +/-41 of sampling noise. A 1% bar sits INSIDE that noise, so
it could not distinguish a real gain from a luckier validation split. +50 clears it.
The bar is deliberately conservative: comparing two models on the same validation pile
is a paired comparison, so true noise on the difference is smaller than +/-41.

**Second reason for a higher bar:** SMOTE carries a cost that is not accuracy. It adds a
dependency, an extra pipeline stage, and a claim that has to be defended -
"the model was trained partly on borrowers who do not exist". That should have to be
bought, not given away.

**Prior:** `class_weight='balanced'` moved the ranking by 0.7% (Spearman 0.9934). SMOTE
is a more invasive form of the same rebalancing idea, so the expectation is that it will
not help either. Recorded here so the prior is on the record rather than claimed
afterwards.

### RESULT — SMOTE REJECTED 2026-09-09

`src/08_smote.py`. SMOTE applied inside an imblearn Pipeline so it runs only during
`fit` and never touches validation, and after preprocessing since it interpolates
between rows and needs numeric input.

| model | defaulters caught @10% | recall | ROC-AUC | fit time |
|---|---|---|---|---|
| baseline | 1,678 | 26.53% | 0.6976 | 2.8s |
| SMOTE | **1,637** | 25.88% | 0.6914 | 7.2s |

**-41 defaulters (-2.44%), against a bar of +50.** The difference sits right at the
+/-41 noise boundary, so the honest statement is "no better, possibly slightly worse".
Either way it fails the pre-registered bar. ROC-AUC also fell, 0.6976 -> 0.6914.

**Not the same story as class_weight, and worth writing up.**

| | rank correlation | same people reviewed |
|---|---|---|
| class_weight | 0.9934 | 91.4% |
| SMOTE | 0.9629 | 79.8% |

Class weighting shouted louder about the same people. SMOTE genuinely REORDERED the
queue - a fifth of the review list changed - and reordered it slightly worse. Plausible
reason: synthetic defaulters are interpolations between real ones, so they occupy
regions of feature space where no borrower actually applied. The model learns a boundary
partly shaped by data that does not exist.

SMOTE also inflated the alarm level the same way class weighting did (mean predicted
probability 3.69% -> 43.43%) while the decision rule still uses only rank.

**DECISION: reject. Baseline stands unchanged.** Both rebalancing approaches have now
been tried and rejected on evidence, which is what the user's original rule required
before moving to tree models.


---

### Stage 10b — PRE-REGISTERED BAR FOR `verification_status`, set 2026-09-10 BEFORE the test

**The finding that prompted this.** `src/09_segments.py` showed the relationship runs
BACKWARDS from intuition:

| verification_status | base default rate | recall | flagged |
|---|---|---|---|
| Verified | 4.97% | 40.6% | 20.5% |
| Source Verified | 3.72% | 21.7% | 8.2% |
| Not Verified | **2.34%** | 7.2% | 1.8% |

Borrowers whose income Lending Club CHECKED default more than twice as often as those
it did not. Model coefficient on `verification_status_Verified` is **+0.32**, one of the
larger weights, pointing toward default.

**Explanation:** Lending Club does not verify at random - it verifies when an application
looks doubtful. The flag records SUSPICION, not reassurance. So the column is partly
encoding Lending Club's own operational triage rather than borrower risk.

**The problem this is (and is not).** Not a fairness problem like
`credit_history_months`. A TRANSFERABILITY problem: a lender with a different
verification policy would see this relationship weaken or reverse. The project's framing
is a general lender routing applicants to review, so a feature encoding one company's
internal process undercuts that.

**The criterion, fixed before any result was seen:**

> Drop `verification_status` if removing it costs FEWER than 50 defaulters at the 10%
> operating point. Keep it, and document the inversion prominently, if removing it costs
> 50 or more.

Note the burden of proof is the reverse of the `credit_history_months` test. There the
feature had to EARN its place against a fairness cost. Here the default action is to keep
unless removal is cheap, because the cost of keeping is milder - a confusing coefficient
and reduced transferability, not unequal treatment of people.

+/-41 sampling noise applies as before, so 50 sits just outside it.

### RESULT — DROPPED 2026-09-10

`src/10_ablation.py verification_status`

| model | caught @10% | recall | ROC-AUC |
|---|---|---|---|
| baseline (all features) | 1,678 | 26.53% | 0.6976 |
| without `verification_status` | 1,663 | 26.29% | 0.6962 |

Cost of removal: **15 defaulters (0.89%)**, inside the +/-41 noise and under the
pre-registered bar of 50. **DROPPED.**

The model no longer depends on one company's internal triage policy, and there is no
counter-intuitive coefficient to explain away. Features 61 -> **60**, categorical 6 -> 5.

Rebuilt, re-split, re-fitted, re-evaluated. New headline figures:

| | before | after |
|---|---|---|
| features | 61 | 60 |
| recall @10% | 26.53% | 26.29% |
| precision @10% | 9.81% | 9.72% |
| ROC-AUC | 0.6976 | 0.6962 |
| Gini | 0.3952 | 0.3923 |
| KS | 29.0 | 29.1 |
| lift @10% | 2.65x | 2.63x |

`src/10_ablation.py` is now a reusable tool - pass it any feature name to test whether
it earns its place.


---

### Stage 10 — Segment analysis — DONE 2026-09-10

`src/09_segments.py`. Asks, for each subgroup: of the defaulters IN THIS GROUP, how many
land in the GLOBAL riskiest 10%? That is the right question because the model ranks
everyone in one list.

**Finding 1 — recall tracks base rate, in every single segment.**

| credit score | base rate | recall | flagged |
|---|---|---|---|
| under 665 | 4.81% | 35.6% | 17.5% |
| 665-695 | 4.11% | 27.9% | 11.9% |
| 695-725 | 2.92% | 18.1% | 5.1% |
| over 725 | 2.14% | 9.1% | 1.7% |

Same shape for income, dti, home ownership, term, employment length, purpose. Not a bug:
a single global ranking concentrates reviews where risk is concentrated. But it IS a
blind spot - only 1.7% of the over-725 group is ever looked at, so 440 of their 484
defaulters sail through. If that segment grows, losses grow invisibly.

**Finding 2 — disparate impact, and the script that tests for it.**

Renters are flagged 15.3% of the time, mortgage holders 5.3% - nearly 3x. Home ownership
is not protected, but tracks age, wealth and (in many countries) ethnicity. Third time
this pattern has appeared: `zip_code`, `credit_history_months`, now `home_ownership`.

`src/09_segments.py` IS a disparate-impact audit - swap the segment for a protected
characteristic and it does what a lender's fairness testing does. We cannot run it that
way because Lending Club collects no demographics. Worth stating in the README: not
collecting a characteristic does not prevent unequal outcomes, it only prevents you
CHECKING for them. Fairness testing requires lawfully collected demographic data.

**Finding 3 — `verification_status` runs backwards.** Led to the ablation above.


---

### Stage 11 — PRE-REGISTERED BANDS FOR TREE MODELS, set 2026-09-10 BEFORE the test

**Why trees might actually help, where the last three did not.** `class_weight`, SMOTE
and rebalancing generally were solving IMBALANCE - a problem the ranking-based decision
rule does not have. Trees solve a real limitation: logistic regression adds weights and
therefore cannot express "it depends". It has one weight for `loan_amnt` and one for
`annual_inc`; it cannot say "loan size matters more when income is low". Trees can,
because each split is conditional on the one above. Lending is full of "it depends", so
this is the first technique tested with a genuine mechanism for improvement.

**The cost, which is NOT accuracy.** Logistic regression gives a coefficient per feature,
so a rejection can be explained per applicant. A forest of hundreds of trees gives a
global importance ranking, not a clean per-applicant reason. That is the **UK GDPR
Article 22** issue in the project's own responsible-use section. It is a heavier cost
than SMOTE's, so the bar must be higher - fairness between tests means matching the bar
to what each option COSTS, not reusing the same number.

**Three bands, fixed before any result was seen.** Baseline is 1,663 defaulters caught
at 10% capacity (60-feature model, after `verification_status` was dropped).

| Result | Verdict |
|---|---|
| under +50 (1,713) | REJECT. Inside the +/-41 sampling noise. |
| +50 to +250 | Real but modest. **Keep logistic regression as the model**, report the tree as a CHALLENGER showing what accuracy exists at a cost we chose not to pay. |
| over +250 (~15%, i.e. 1,913+) | Large enough that giving up per-applicant explanations becomes a genuine argument. Revisit the deployment choice. |

The floor (+50) is principled: ~+/-41 is measurable sampling noise. Everything above it
is a value judgement about how much explainability is worth, and is stated as such
rather than dressed up as arithmetic.

**The middle band mirrors real practice.** Lenders commonly build a GBM challenger
alongside the scorecard to see what accuracy is being left on the table, and still deploy
the scorecard, because explainability obligations outrank a few points of Gini.

**Honest methodology note.** XGBoost uses early stopping, which needs a held-out set. If
it early-stopped on the validation pile, that pile would have influenced the model and
the comparison would flatter XGBoost. So 15% is carved out of TRAIN for early stopping
and validation stays clean. Random Forest needs no equivalent.

**RESULT — 16/09/2026.**

```
model                      caught   recall   ROC-AUC    Gini   vs baseline
logistic (baseline)         1,663   26.29%    0.6962  0.3923
random forest               1,630   25.77%    0.6979  0.3957          -33   REJECT
xgboost                     1,785   28.22%    0.7161  0.4322         +122   CHALLENGER
```

Validation. XGBoost lands at **+122**, inside the pre-registered middle band
(+50 to +250). Verdict as written before the test: **keep logistic regression as the
model, report XGBoost as a challenger.** Random forest falls below the +50 floor and is
rejected — it is inside sampling noise.

No band was moved and no threshold was reinterpreted after seeing the number.

---

## Stage 13 — the sealed test set, opened once. 16/09/2026

Model selection closed with the tree result above. Validation had by then been used for
four decisions (stages 7, 9, 10, 11), so it was no longer an innocent estimate. Test was
scored once, with nothing downstream changed as a result.

```
                      validation       TEST        gap
logistic  Gini            0.3923     0.3831    -0.0092
          ROC-AUC         0.6962     0.6916    -0.0046
          KS                29.1       27.5       -1.6
          caught @10%      1,663      1,663         +0

xgboost   Gini            0.4322     0.4273    -0.0049
          ROC-AUC         0.7161     0.7137    -0.0024
          caught @10%      1,785      1,789         +4
```

About one point of Gini lost between validation and test. That gap is the size of the
optimism the four validation-based decisions introduced, and it is small — the discipline
of fitting medians inside the Pipeline and carving early-stopping out of train held up.

**The identical count is a coincidence, and was checked.** Both piles hold exactly
171,101 rows and exactly 6,326 defaulters, and the model caught 1,663 in each at the 10%
cut. The predictions differ (mean 0.036963 vs 0.036953) and AUC moves, so it is one
coincidence rather than four — all confusion-matrix cells follow from that single number.

**Calibration, checked cheaply.** Predicted vs observed by decile on test: ratio 0.93–1.05
through deciles 1–8, drifting to 1.12 and 1.15 in the two safest. Well calibrated where
the decision is made; over-predicts among the safest applicants. Fine for ranking, not
fine for IFRS 9 ECL without recalibration. A reliability curve and Brier score would be
the proper test.

**An unseen category surfaced.** `purpose = 'educational'` appears in test but not in
train, and is encoded as all zeros by `handle_unknown='ignore'`. Harmless at this
frequency; in production it is exactly what a monitoring rule should catch.

On the challenger: +126 defaulters on test (+122 on validation) — the pre-registered
verdict holds out of sample, which is the useful part.
