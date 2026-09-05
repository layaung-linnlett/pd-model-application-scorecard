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
Open questions put to the user on 2026-09-05:
1. Where does the Lending Club file come from / where will it live?
2. How is `default_within_12_months` actually constructed, given that the accepted-loans
   file records only a single snapshot `loan_status`, not a month-by-month payment history?

Not yet decided. Nothing downstream may start.

---

## Explicitly agreed / approved

| Date | Decision |
|------|----------|
| 2026-09-05 | Project framing, target concept, and session discipline as stated by user |
| 2026-09-05 | Repo scaffold and tooling conventions |

## Pending (not started)

- [ ] Stage 1 — Data source confirmed; target definition agreed and implemented
- [ ] Stage 2 — Leakage checklist (application-time vs post-origination columns)  **GATE**
- [ ] Stage 3 — Stratified train/validation/test split  **GATE**
- [ ] Stage 4 — Logistic regression baseline + coefficient walkthrough  **GATE**
- [ ] Stage 5 — Evaluation: confusion matrix, precision, recall, ROC-AUC, KS, Gini  **GATE**
- [ ] Stage 6 — Later models (only after baseline approved; `class_weight='balanced'` before SMOTE)  **GATE**
- [ ] Stage 7 — README: business framing, leakage checklist, missing-data policy,
      Responsible-use and limitations (FCA CONC 5.2A, UK GDPR Art. 22, Equality Act 2010)
