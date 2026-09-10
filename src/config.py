"""Column policy for the PD application scorecard.

The single rule: a column may be a feature only if the lender would have known
its value ON THE DAY THE BORROWER APPLIED. Everything else is leakage.

Decisions recorded here were agreed with the user; see docs/leakage_checklist.md
for the reasoning and PROGRESS.md for the dates.
"""

# --- Columns used to BUILD THE LABEL. Never features. ---
LABEL_SOURCE = [
    "issue_d",        # origination date: cohort filter + start of the 12-month window
    "loan_status",    # terminal status: the Charged Off / Default test
    "last_pymnt_d",   # last payment: drives the 9-month cutoff (Option B)
]

# --- Application-time features: KEEP ---
# Borrower-stated on the application form
APPLICATION_STATED = [
    "loan_amnt", "term", "purpose", "application_type",
    "emp_length", "home_ownership", "annual_inc",
    "dti",
]

# Credit-bureau attributes pulled at application
BUREAU_CORE = [
    "fico_range_low", "fico_range_high",
    "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc",
    "collections_12_mths_ex_med", "mths_since_last_delinq",
    "mths_since_last_record", "mths_since_last_major_derog",
    "acc_now_delinq", "delinq_amnt", "pub_rec_bankruptcies", "tax_liens",
    "chargeoff_within_12_mths",
]

BUREAU_EXTENDED = ["tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim",
     "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy", "bc_util", "mo_sin_rcnt_rev_tl_op",
     "mo_sin_rcnt_tl", "mort_acc", "mths_since_recent_bc",
     "mths_since_recent_bc_dlq", "mths_since_recent_inq",
     "mths_since_recent_revol_delinq", "num_accts_ever_120_pd",
     "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats", "num_bc_tl",
     "num_il_tl", "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
     "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
     "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75",
     "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
     "total_il_high_credit_limit",
]

# 6. Credit-bureau fields Lending Club only began collecting in Dec 2015.
#    NOT a timing leak and NOT ordinary missingness: they are missing for ~100%
#    of 2015 originations and ~0% of 2016 ones, so their presence encodes WHEN
#    the loan was issued rather than anything about the borrower. A model given
#    these (or missing-indicators for them) would learn "issued before Dec 2015
#    = safer", which is a vintage artefact with no predictive value for a new
#    applicant. Dropped by user decision 2026-09-07.
#    `il_util` and `mths_since_rcnt_il` are mixed cases - still 13.2% and 2.7%
#    missing in 2016, for borrowers with no installment loans - but they carry
#    the same Dec-2015 discontinuity, so they go with the block.
DROP_TIME_VARYING_AVAILABILITY = [
    "all_util", "inq_last_12m", "total_cu_tl", "open_acc_6m", "open_il_24m",
    "open_act_il", "open_il_12m", "max_bal_bc", "open_rv_12m", "open_rv_24m",
    "inq_fi", "total_bal_il", "il_util", "mths_since_rcnt_il",
]

FEATURES = APPLICATION_STATED + BUREAU_CORE + BUREAU_EXTENDED

# --- DROPPED ---

# 1. Post-origination facts. Classic leakage: written down after the decision.
DROP_LEAKAGE = [
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_amnt",
    "next_pymnt_d", "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low",   # note: "last_" = post-origination
    "pymnt_plan", "disbursement_method", "initial_list_status",
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
    "deferral_term", "hardship_amount", "hardship_start_date",
    "hardship_end_date", "payment_plan_start_date", "hardship_length",
    "hardship_dpd", "hardship_loan_status",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    "debt_settlement_flag", "debt_settlement_flag_date",
    "settlement_status", "settlement_date", "settlement_amount",
    "settlement_percentage", "settlement_term",
]

# 2. Lending Club's own risk assessment. Not a timing leak - a "copying the
#    lender's homework" problem. Dropped by user decision 2026-09-06.
#    `installment` is included because installment = f(loan_amnt, term, int_rate),
#    so keeping it would smuggle int_rate back in.
DROP_LENDER_JUDGEMENT = ["int_rate", "grade", "sub_grade", "installment"]

# 3. Identifiers, free text, constants, and near-duplicates of loan_amnt.
DROP_NON_PREDICTIVE = [
    "id", "member_id", "url", "policy_code",
    "desc", "emp_title", "title",              # free text, not modelled in baseline
    "funded_amnt", "funded_amnt_inv",          # post-decision duplicates of loan_amnt
]

# 4. Joint-application and secondary-applicant fields. Application-time, but
#    populated only for the small joint-application minority in 2015-2016.
DROP_JOINT_SPARSE = [
    "annual_inc_joint", "dti_joint", "verification_status_joint",
    "revol_bal_joint",
    "sec_app_fico_range_low", "sec_app_fico_range_high",
    "sec_app_earliest_cr_line", "sec_app_inq_last_6mths", "sec_app_mort_acc",
    "sec_app_open_acc", "sec_app_revol_util", "sec_app_open_act_il",
    "sec_app_num_rev_accts", "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med",
    "sec_app_mths_since_last_major_derog",
]

# 5. Geography. Known at application, so NOT a timing leak. Dropped on
#    fair-lending grounds by user decision 2026-09-06: location is a proxy for
#    protected characteristics under the Equality Act 2010, so a model using it
#    can reproduce discrimination without ever naming the characteristic.
DROP_FAIRNESS_PROXY = ["zip_code", "addr_state"]

COHORT_START, COHORT_END = "2015-01", "2016-12"


# ---------------------------------------------------------------------------
# Missing-data policy (agreed 2026-09-07). See docs/missing_data_policy.md.
#
# Rule 1  numeric column with blanks -> add a binary "<col>_missing" flag, then
#         fill the blank with the MEDIAN OF THE TRAINING PILE.
# Rule 2  emp_length is text; blank becomes its own category, "Unknown".
# Rule 3  every fill value is computed on train only and applied unchanged to
#         val and test. Computing it across all the data would leak.
#
# Flags are added only where at least 0.5% of training rows are blank. Below
# that there is too little data for the flag to carry meaning.
# ---------------------------------------------------------------------------

MISSING_FLAG_COLUMNS = [
    "mths_since_last_record",          # 81.6% blank - no public record
    "mths_since_recent_bc_dlq",        # 74.3% - no bankcard delinquency
    "mths_since_last_major_derog",     # 70.9% - no major derogatory event
    "mths_since_recent_revol_delinq",  # 63.8% - no revolving delinquency
    "mths_since_last_delinq",          # 47.8% - never delinquent
    "mths_since_recent_inq",           # 10.5% - no recent credit enquiry
    "num_tl_120dpd_2m",                #  4.8%
    "bc_util",                         #  1.1%
    "percent_bc_gt_75",                #  1.0%
    "bc_open_to_buy",                  #  1.0%
    "mths_since_recent_bc",            #  0.9%
]

# Median-filled but NOT flagged: too few blanks for a flag to mean anything.
MEDIAN_FILL_ONLY = ["dti", "revol_util", "inq_last_6mths", "num_rev_accts"]

# Text features. `emp_length` carries an informative blank -> "Unknown".
CATEGORICAL_FEATURES = [
    "term", "purpose", "application_type", "emp_length",
    "home_ownership",
]

# 7. Age proxies. Dropped 2026-09-07 after a pre-registered test.
#
# `earliest_cr_line` was previously converted into `credit_history_months`
# (months of credit history at application). That value places a floor under
# the borrower's AGE across its whole range - 600 months of history can only
# belong to someone around 68 - and age is protected under the Equality Act 2010.
#
# `mo_sin_old_rev_tl_op` correlates with it at 0.92 and `mo_sin_old_il_acct` at
# 0.35; all three measure "how long ago did you start using credit". Dropping
# only one would have been cosmetic, because the other two carry the same signal.
#
# The test (validation pile, riskiest 5% reviewed, 6,326 actual defaulters):
#     keep everything            964 defaulters caught
#     drop credit_history_months 972
#     drop the 0.92 pair         960
#     drop all age-linked        961
# All within the +/-31 expected from chance, so removing them costs nothing
# measurable. The pre-registered bar was a 1% relative gain to justify KEEPING
# a feature with known fairness risk; it was not met.
#
# NOTE: this does NOT make the model age-blind. Account counts (`total_acc`,
# `num_rev_accts`, `mort_acc`) still correlate with age at 0.26-0.31. They are
# retained as legitimately distinct measures, and the residual is disclosed in
# the README rather than hidden.
DROP_AGE_PROXY = [
    "earliest_cr_line", "mo_sin_old_rev_tl_op", "mo_sin_old_il_acct",
]

# 8. The lender's own operational process. Dropped 2026-09-10 after a
#    pre-registered test (commit 6631598).
#
#    `verification_status` runs BACKWARDS from intuition: borrowers whose income
#    Lending Club checked default at 4.97%, those it did not at 2.34%. The model
#    gave it +0.32, one of the larger weights, pointing toward default.
#
#    Reason: Lending Club does not verify at random - it verifies when an
#    application looks doubtful. The flag records SUSPICION, not reassurance. So
#    the column encodes Lending Club's triage policy rather than borrower risk,
#    and a lender with a different policy would see it weaken or reverse.
#
#    Removing it cost 15 defaulters at the 10% operating point, inside the +/-41
#    sampling noise and under the pre-registered bar of 50. Dropped: the model no
#    longer depends on one company's internal process, and there is no
#    counter-intuitive coefficient to explain away.
DROP_LENDER_PROCESS = ["verification_status"]

# No derived features remain now that credit_history_months is gone. The hooks
# stay so the inventory tooling keeps working.
DATE_FEATURES_TO_DERIVE: list[str] = []
DERIVED_FEATURES: list[str] = []


def model_features() -> list[str]:
    """The columns actually handed to the model."""
    return [c for c in FEATURES if c not in DATE_FEATURES_TO_DERIVE] + DERIVED_FEATURES


# ---------------------------------------------------------------------------
# Data-quality limits (agreed 2026-09-07). See docs/data_quality_decision_log.md
#
# Values outside these bounds are set to MISSING, not deleted and not capped:
#   - not deleted, because one bad field should not discard a borrower's other
#     60 valid columns;
#   - not capped, because capping asserts a value we do not believe. Setting to
#     missing says honestly "we do not know", and hands the row to the
#     missing-data machinery already agreed in docs/missing_data_policy.md.
#
# These bounds target IMPOSSIBLE values only. Rare-but-real extremes are kept
# deliberately - the evidence says they carry signal (dti 39-100 defaults at
# 6.41% against a 3.70% base rate; revol_util 100-150 at 4.66%).
# ---------------------------------------------------------------------------

DATA_QUALITY_LIMITS = {
    # 99 borrowers state income under $5,000. 93 of them borrowed MORE than
    # their stated annual income and 72 were never income-verified. The figure
    # is not credible, and it is what makes their dti explode.
    "annual_inc": {"min": 5_000},

    # dti = debt / income, so a broken income produces an absurd dti. Risk rises
    # with dti up to ~39 (6.41% default) then COLLAPSES above 100 (1.23%) -
    # real financial stress does not behave that way. 999 is a placeholder.
    "dti": {"min": 0, "max": 100},

    # Utilisation above 100% is real (people go over their limit) and predictive.
    # Above 150% is not credible.
    "revol_util": {"max": 150},

    # 9,999,999 is an all-9s placeholder, not a credit limit.
    "total_rev_hi_lim": {"max": 9_999_998},

    # 61 bankruptcies/liens for one borrower is not credible. Safe to clear
    # because the dose-response is flat: the signal is "has any public record",
    # and the count carries nothing beyond the first.
    "pub_rec": {"max": 20},
    "tax_liens": {"max": 20},
}


# ---------------------------------------------------------------------------
# Operating point (set 2026-09-09).
#
# The model outputs a RANKED LIST of all applicants, riskiest first. It does not
# decide who to review - the lender's staffing does. This constant is how far
# down that list we read.
#
# 10% is an ASSUMPTION, not a fact about any real lender. It is stated here so
# every script uses the same figure and so the assumption is visible rather than
# buried. The full curve is reported alongside it in the README, because the
# trade-off is more informative than any single number:
#
#     review  1%  ->  catch  3.8% of defaulters   (3.8x better than random)
#     review  5%  ->  catch 15.3%                 (3.1x)
#     review 10%  ->  catch 26.5%                 (2.7x)   <- stated operating point
#     review 20%  ->  catch 43.9%                 (2.2x)
#     review 50%  ->  catch 76.1%                 (1.5x)
#
# A real deployment would set this from the cost of a review (~1 analyst hour)
# against the cost of a default (a ~$14k loan, partly recovered). That arithmetic
# needs inputs this dataset does not contain - how often review actually prevents
# a bad loan, and what is lost by wrongly rejecting a good applicant - so it is
# not attempted here.
# ---------------------------------------------------------------------------

REVIEW_CAPACITY = 0.10
