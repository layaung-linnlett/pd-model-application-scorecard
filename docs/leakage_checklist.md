# Leakage checklist

**The rule:** a column may be used as a feature only if the lender would have known
its value **on the day the borrower applied**.

**Why it matters.** If a column records something that only became true after the loan
was granted, the model looks excellent in testing and then fails in production — because
that column does not exist when a real applicant is sitting in front of you. It is the
equivalent of predicting tomorrow's rain from a photograph of tomorrow's wet pavement.

Source file: `accepted_2007_to_2018Q4.csv` (151 columns).
Policy implemented in [`src/config.py`](../src/config.py).

| Outcome | Columns |
|---|---:|
| Kept as features | 78 |
| Used to build the label (never features) | 3 |
| Dropped | 70 |

---

## Kept — known at application time (78)

**Stated by the borrower on the form (9)**
`loan_amnt`, `term`, `purpose`, `application_type`, `emp_length`, `home_ownership`,
`annual_inc`, `verification_status`, `dti`

**Credit-bureau attributes pulled at application (69)**
Core: `fico_range_low`, `fico_range_high`, `earliest_cr_line`, `delinq_2yrs`,
`inq_last_6mths`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc`,
`collections_12_mths_ex_med`, `mths_since_last_delinq`, `mths_since_last_record`,
`mths_since_last_major_derog`, `acc_now_delinq`, `delinq_amnt`, `pub_rec_bankruptcies`,
`tax_liens`, `chargeoff_within_12_mths`

Extended bureau panel: the `num_*`, `mo_sin_*`, `mths_since_recent_*`, `open_*`,
`total_*`, `bc_*`, `il_util`, `all_util`, `avg_cur_bal`, `mort_acc`, `pct_tl_nvr_dlq`,
`percent_bc_gt_75` families — see `BUREAU_EXTENDED` in `src/config.py`.

---

## Used only to build the label (3)

`issue_d`, `loan_status`, `last_pymnt_d`

These define `default_within_12_months`. They are never fed to the model as features.

---

## Dropped — post-origination leakage (39)

Everything recorded after the money changed hands: repayment totals
(`total_pymnt`, `total_rec_prncp`, `total_rec_int`, `out_prncp`, …), recovery and
collection amounts (`recoveries`, `collection_recovery_fee`), payment dates
(`last_pymnt_amnt`, `next_pymnt_d`, `last_credit_pull_d`), the entire `hardship_*`
block, and the entire `settlement_*` / `debt_settlement_*` block.

**Two named traps worth remembering:**

- **`last_fico_range_high` / `last_fico_range_low` — LEAK.** Sounds like a credit
  score, and credit scores are legitimate. But the prefix `last_` means *most recently
  pulled*, and those pulls continued until the April 2019 snapshot. The application-time
  score is `fico_range_high` / `fico_range_low`, without the prefix. Same concept,
  two different moments; only one is allowed.
- **`chargeoff_within_12_mths` — SAFE.** Sounds like the target. It is not. It counts
  charge-offs on the borrower's *other* credit accounts, reported by the bureau
  *before* they applied.

The lesson: a column name cannot settle this. Only the question "when was this value
written down?" can.

---

## Dropped — Lending Club's own risk assessment (4)

`int_rate`, `grade`, `sub_grade`, `installment`

Not a timing leak — these are all known at application. The problem is different:
Lending Club already ran its own risk model and expressed the answer as a grade and a
price. Feeding those in means the model partly copies Lending Club's homework instead
of assessing risk from borrower characteristics.

`installment` is dropped for a subtler reason: `installment = f(loan_amnt, term, int_rate)`,
so keeping it would smuggle the interest rate back in through the side door. **Dropping
a leaking column is not enough — anything that is a function of it must go too.**

These may be reintroduced later, deliberately, as a *benchmark* to see how close the
scorecard gets to Lending Club's own judgement. That is a separate experiment, not
part of the model.

---

## Dropped — fair-lending proxies (2)

`zip_code`, `addr_state`

Known at application, so not leakage. Dropped on fairness grounds: location correlates
strongly with protected characteristics under the **Equality Act 2010**, so a model
using it can reproduce discrimination without the protected characteristic ever
appearing in the data. In US lending the historical practice has a name — redlining.

The small loss of predictive power is an accepted cost. See the Responsible-use
section of the README.

---

## Dropped — identifiers, free text, duplicates (9)

`id`, `member_id`, `url`, `policy_code` (constant), `desc`, `emp_title`, `title`
(free text, not modelled in this baseline), `funded_amnt`, `funded_amnt_inv`
(post-decision near-duplicates of `loan_amnt`).

## Dropped — joint-application fields (16)

`annual_inc_joint`, `dti_joint`, `verification_status_joint`, `revol_bal_joint`, and
the twelve `sec_app_*` columns. These are application-time and legitimate, but are
populated only for the small joint-application minority in 2015–2016, so they are
almost entirely missing. Excluded from the baseline for simplicity.
