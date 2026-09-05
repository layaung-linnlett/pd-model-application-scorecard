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

**Stage 1 — Data source & target definition.** BLOCKED, awaiting user decision.

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

**Still open 2026-09-05:** which Kaggle dataset to download.
Claude recommends `wordsforthewise/lending-club` (2007-2018Q4) over the classic
`wendykan/lending-club-loan-data` (2007-2015), because the 2015-cutoff snapshot
right-censors the 2015-2016 cohort and would systematically mislabel Option B defaults
as non-defaults. Awaiting user decision. Nothing downstream may start.

---

## Explicitly agreed / approved

| Date | Decision |
|------|----------|
| 2026-09-05 | Project framing, target concept, and session discipline as stated by user |
| 2026-09-05 | Repo scaffold and tooling conventions |
| 2026-09-05 | Target definition: **Option B**, reconstructed 12-month window via `last_pymnt_d`, 9-month cutoff, sensitivity test required |

## Pending (not started)

- [ ] Stage 1 — Data source confirmed; target definition agreed and implemented
- [ ] Stage 2 — Leakage checklist (application-time vs post-origination columns)  **GATE**
- [ ] Stage 3 — Stratified train/validation/test split  **GATE**
- [ ] Stage 4 — Logistic regression baseline + coefficient walkthrough  **GATE**
- [ ] Stage 5 — Evaluation: confusion matrix, precision, recall, ROC-AUC, KS, Gini  **GATE**
- [ ] Stage 6 — Later models (only after baseline approved; `class_weight='balanced'` before SMOTE)  **GATE**
- [ ] Stage 7 — README: business framing, leakage checklist, missing-data policy,
      Responsible-use and limitations (FCA CONC 5.2A, UK GDPR Art. 22, Equality Act 2010)
