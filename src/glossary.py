"""Plain-English meaning of every column in the Lending Club file.

These are written for a reader meeting the data for the first time, not copied
from Lending Club's official data dictionary (`LCDataDictionary.xlsx`, shipped
with the Kaggle download). If you want the official wording, drop that file into
`data/raw/` and cross-check.

Abbreviations that appear throughout the column names:

    bc      bankcard, i.e. a credit card
    il      installment loan - fixed monthly payments, e.g. a car loan
    rev/rv  revolving credit - credit cards and similar, where the balance moves
    tl      trade line - any single account on a credit file
    sats    satisfactory - accounts in good standing
    dpd     days past due
    derog   derogatory - a serious black mark
    delinq  delinquent - behind on payments
    util    utilisation - how much of a credit limit is being used
    inq     inquiry - a credit check by a lender
    pub_rec public record - bankruptcy, tax lien or court judgment
    mo_sin  months since
"""

MEANING = {
    # --- the loan itself ---
    "id": "Unique reference number for the loan",
    "member_id": "Unique reference for the borrower (empty in this file)",
    "loan_amnt": "How much the borrower asked to borrow",
    "funded_amnt": "How much was actually lent",
    "funded_amnt_inv": "The share of the loan funded by investors",
    "term": "Length of the loan: 36 or 60 months",
    "int_rate": "Interest rate charged",
    "installment": "The fixed monthly payment",
    "grade": "Lending Club's own risk grade, A (safest) to G (riskiest)",
    "sub_grade": "Finer version of the grade, e.g. B3",
    "issue_d": "The month the loan was granted",
    "loan_status": "Where the loan stood at the snapshot: Fully Paid, Current, Charged Off...",
    "purpose": "What the loan is for: debt consolidation, car, home improvement...",
    "title": "Loan title the borrower typed in themselves",
    "desc": "Free-text description the borrower wrote",
    "url": "Web link to the loan's page",
    "policy_code": "Always 1 in this file - carries no information",
    "initial_list_status": "Whether the loan was listed to investors whole or in fractions",
    "disbursement_method": "How the money was paid out (cash, or direct to creditors)",
    "application_type": "Individual application, or joint with a second person",
    "pymnt_plan": "Whether the loan was put on a special payment plan",

    # --- who the borrower is ---
    "emp_title": "Job title the borrower typed in",
    "emp_length": "How long they have been employed, 0 to 10+ years",
    "home_ownership": "Do they rent, own outright, or have a mortgage",
    "annual_inc": "Yearly income the borrower stated",
    "verification_status": "Whether Lending Club checked that income was real",
    "zip_code": "First three digits of their postcode",
    "addr_state": "Which US state they live in",
    "dti": "Debt-to-income: monthly debt payments as a % of monthly income",

    # --- credit history at application ---
    "earliest_cr_line": "The month they opened their very first credit account ever",
    "fico_range_low": "Lower end of their credit score band when they applied",
    "fico_range_high": "Upper end of their credit score band when they applied",
    "delinq_2yrs": "Times they fell 30+ days behind in the past 2 years",
    "inq_last_6mths": "Credit checks by lenders in the last 6 months",
    "mths_since_last_delinq": "Months since they were last behind on a payment (blank = never)",
    "mths_since_last_record": "Months since their last public record (blank = none)",
    "mths_since_last_major_derog": "Months since a serious black mark, 90+ days late (blank = none)",
    "open_acc": "Credit accounts currently open",
    "pub_rec": "Number of bankruptcies, tax liens or judgments on record",
    "revol_bal": "Total balance owed on credit cards and similar",
    "revol_util": "% of their credit card limits currently used",
    "total_acc": "Total credit accounts ever opened",
    "collections_12_mths_ex_med": "Debts passed to collectors in the last year (excluding medical)",
    "acc_now_delinq": "Accounts currently behind on payments",
    "delinq_amnt": "Amount currently overdue",
    "pub_rec_bankruptcies": "Number of bankruptcies on record",
    "tax_liens": "Number of unpaid-tax claims against them",
    "chargeoff_within_12_mths": "Write-offs on their OTHER accounts in the last year",
    "tot_coll_amt": "Total ever passed to debt collectors",
    "tot_cur_bal": "Total balance across all their accounts",

    # --- bureau detail: accounts and balances ---
    "acc_open_past_24mths": "Accounts opened in the last 2 years",
    "avg_cur_bal": "Average balance across their accounts",
    "bc_open_to_buy": "Credit card headroom left: limit minus balance",
    "bc_util": "% of their credit card limits used",
    "mo_sin_old_il_acct": "Months since their oldest installment loan was opened",
    "mo_sin_old_rev_tl_op": "Months since their oldest credit card was opened",
    "mo_sin_rcnt_rev_tl_op": "Months since their newest credit card was opened",
    "mo_sin_rcnt_tl": "Months since their newest account of any kind",
    "mort_acc": "Number of mortgage accounts",
    "mths_since_recent_bc": "Months since their newest credit card was opened",
    "mths_since_recent_bc_dlq": "Months since last late on a credit card (blank = never)",
    "mths_since_recent_inq": "Months since a lender last checked their credit (blank = none recently)",
    "mths_since_recent_revol_delinq": "Months since last late on a credit card or line (blank = never)",
    "num_accts_ever_120_pd": "Accounts ever 120+ days late",
    "num_actv_bc_tl": "Credit cards currently in use",
    "num_actv_rev_tl": "Revolving accounts currently in use",
    "num_bc_sats": "Credit cards in good standing",
    "num_bc_tl": "Total credit cards",
    "num_il_tl": "Total installment loans",
    "num_op_rev_tl": "Revolving accounts currently open",
    "num_rev_accts": "Revolving accounts ever opened",
    "num_rev_tl_bal_gt_0": "Revolving accounts carrying a balance above zero",
    "num_sats": "Accounts in good standing",
    "num_tl_120dpd_2m": "Accounts 120+ days late in the last 2 months",
    "num_tl_30dpd": "Accounts 30 days late right now",
    "num_tl_90g_dpd_24m": "Accounts 90+ days late at some point in the last 2 years",
    "num_tl_op_past_12m": "Accounts opened in the last year",
    "pct_tl_nvr_dlq": "% of their accounts that have never been late",
    "percent_bc_gt_75": "% of their credit cards charged above 75% of the limit",
    "tot_hi_cred_lim": "Total credit limit across everything",
    "total_bal_ex_mort": "Total owed, excluding mortgages",
    "total_bc_limit": "Total credit card limit",
    "total_il_high_credit_limit": "Total installment loan limit",
    "total_rev_hi_lim": "Total revolving credit limit",

    # --- bureau block only collected from Dec 2015 ---
    "open_acc_6m": "Accounts opened in the last 6 months",
    "open_act_il": "Installment loans currently active",
    "open_il_12m": "Installment loans opened in the last year",
    "open_il_24m": "Installment loans opened in the last 2 years",
    "mths_since_rcnt_il": "Months since their most recent installment loan",
    "total_bal_il": "Total owed on installment loans",
    "il_util": "% of installment loan limits used",
    "open_rv_12m": "Revolving accounts opened in the last year",
    "open_rv_24m": "Revolving accounts opened in the last 2 years",
    "max_bal_bc": "Highest balance sitting on any single credit card",
    "all_util": "% of all their credit limits used",
    "inq_fi": "Number of personal-finance credit checks",
    "total_cu_tl": "Number of credit union accounts",
    "inq_last_12m": "Credit checks by lenders in the last year",

    # --- what happened AFTER the loan was granted (leakage) ---
    "out_prncp": "Loan amount still outstanding",
    "out_prncp_inv": "Amount still outstanding, investor share",
    "total_pymnt": "Total repaid so far",
    "total_pymnt_inv": "Total repaid so far, investor share",
    "total_rec_prncp": "How much of the original loan has been repaid",
    "total_rec_int": "Interest paid so far",
    "total_rec_late_fee": "Late fees paid",
    "recoveries": "Money clawed back after the debt was written off",
    "collection_recovery_fee": "Fee charged on money clawed back",
    "last_pymnt_d": "Month of their most recent payment",
    "last_pymnt_amnt": "Size of their most recent payment",
    "next_pymnt_d": "Month the next payment is due",
    "last_credit_pull_d": "Month Lending Club last checked their credit",
    "last_fico_range_high": "Most recent credit score, upper end - updated AFTER the loan started",
    "last_fico_range_low": "Most recent credit score, lower end - updated AFTER the loan started",
    "debt_settlement_flag": "Whether they settled the debt for less than owed",
    "debt_settlement_flag_date": "When that settlement was flagged",
    "settlement_status": "Status of the settlement agreement",
    "settlement_date": "When the settlement was agreed",
    "settlement_amount": "Amount they agreed to pay",
    "settlement_percentage": "That amount as a % of what was owed",
    "settlement_term": "How many months the settlement runs",

    # --- hardship plans (all post-origination) ---
    "hardship_flag": "Whether the borrower was put on a hardship plan",
    "hardship_type": "Type of hardship plan",
    "hardship_reason": "Reason given for the hardship",
    "hardship_status": "Whether the plan is active, completed or broken",
    "deferral_term": "Months of payment deferred under the plan",
    "hardship_amount": "Reduced payment under the plan",
    "hardship_start_date": "When the plan started",
    "hardship_end_date": "When the plan ended",
    "payment_plan_start_date": "When the payment plan began",
    "hardship_length": "Length of the plan in months",
    "hardship_dpd": "Days past due when the plan started",
    "hardship_loan_status": "Loan status when the plan started",
    "orig_projected_additional_accrued_interest": "Extra interest expected because of the plan",
    "hardship_payoff_balance_amount": "Balance owed at the start of the plan",
    "hardship_last_payment_amount": "Last payment made before the plan",

    # --- joint applications and the second applicant ---
    "annual_inc_joint": "Combined yearly income of both applicants",
    "dti_joint": "Combined debt-to-income of both applicants",
    "verification_status_joint": "Whether the combined income was checked",
    "revol_bal_joint": "Combined credit card balance of both applicants",
    "sec_app_fico_range_low": "Second applicant's credit score, lower end",
    "sec_app_fico_range_high": "Second applicant's credit score, upper end",
    "sec_app_earliest_cr_line": "Second applicant's first ever credit account",
    "sec_app_inq_last_6mths": "Second applicant's credit checks in last 6 months",
    "sec_app_mort_acc": "Second applicant's mortgage accounts",
    "sec_app_open_acc": "Second applicant's open accounts",
    "sec_app_revol_util": "Second applicant's credit card utilisation",
    "sec_app_open_act_il": "Second applicant's active installment loans",
    "sec_app_num_rev_accts": "Second applicant's revolving accounts",
    "sec_app_chargeoff_within_12_mths": "Second applicant's write-offs in the last year",
    "sec_app_collections_12_mths_ex_med": "Second applicant's debts sent to collectors",
    "sec_app_mths_since_last_major_derog": "Second applicant's months since a serious black mark",
}
