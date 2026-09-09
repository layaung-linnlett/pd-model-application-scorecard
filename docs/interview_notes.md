# Interview notes — PD application scorecard

Concepts worked through while building this project, with the actual numbers, and
the sentence to say out loud.

Every figure here comes from the project's own output. Nothing is invented.
If a number changes, change it here too.

**The project in one line:** using only what a lender knows on the day someone
applies, estimate their chance of defaulting within 12 months, so the riskiest
applicants get manual review instead of everyone.

**The headline numbers:**

```
855,502 Lending Club loans, originated 2015-2016
3.70% defaulted within 12 months          26 : 1 imbalance
61 features, all known at application time
ROC-AUC 0.6976 · Gini 0.3952 · KS 29.0
at 10% review capacity: 1,678 of 6,326 defaulters caught (26.5%), 2.65x random
```

---

## 1. Leakage — the test is *when*, not *what*

A column is only allowed if the lender would have known its value **on the day of
the application**. 84 of the 151 raw columns failed that test.

**The wet pavement.** Predicting tomorrow's rain from a photograph of tomorrow's wet
pavement. Perfect in testing, useless tomorrow morning, because the photo won't exist.

**Three traps in this dataset:**

| Column | Verdict | Why |
|---|---|---|
| `fico_range_high` | safe | credit score at application |
| `last_fico_range_high` | **leaks** | `last_` = refreshed after the loan started |
| `chargeoff_within_12_mths` | **safe** | sounds like the target; actually the borrower's *other* accounts, before applying |
| `installment` | dropped | = f(loan_amnt, term, **int_rate**) — smuggles the interest rate back in |

**Say it like this:**
> "The name can't settle it — `last_fico_range_high` sounds legitimate and leaks,
> `chargeoff_within_12_mths` sounds like the target and doesn't. The only question
> that works is *when was this value written down*."

**The subtler lesson:** dropping a leaking column isn't enough. Anything **derived
from it** has to go too — which is why `installment` went with `int_rate`.

---

## 2. Why accuracy is banned here

At a 3.70% default rate, a model that predicts "nobody ever defaults" scores
**96.3% accuracy** and is worth nothing.

**Say it like this:**
> "Accuracy is meaningless at a 26:1 imbalance. I reported recall, precision,
> ROC-AUC, Gini and KS instead — every one of them survives the imbalance."

---

## 3. Precision vs recall, and why low precision is fine *here*

Both come from the **same cell** of the confusion matrix, divided differently.

```
                      DID default    did NOT
REVIEWED                    1,678     15,432
approved unreviewed         4,648    149,343

recall     1,678 ÷ 6,326   = 26.5%   read DOWN the column
precision  1,678 ÷ 17,110  =  9.8%   read ACROSS the row
```

Precision of 9.8% means 9 in 10 reviews find nothing. That sounds bad until you
price the two mistakes:

```
wasted review      ~£40      one analyst hour
missed defaulter   ~£7,000   a charged-off loan
                             roughly 175 : 1
```

**Say it like this:**
> "It's not 9 wasted out of 10 — it's 9 wasted instead of 26, because random
> selection finds one defaulter in 27. And a false alarm costs an analyst hour
> against thousands for a missed default, so I can afford a lot of them.
> Precision only matters relative to what a false alarm costs. If a 'review' were
> an invasive medical procedure, 9 in 10 unnecessary ones would be a scandal."

---

## 4. The 0.5 threshold is not a decision anybody made

The baseline flagged **2 borrowers out of 171,101** at a 0.5 cut-off. That looked
like failure. It wasn't.

Only 3.7% of borrowers default, so the model is rarely more than 50% sure about
anyone — its single highest score in 171,101 people is 56%. That's the model being
**honest**, not broken. 0.5 is just scikit-learn's default.

**The dry-country forecaster.** It rains four days a year, so the forecaster never
says "70% chance". If your rule is "take an umbrella above 50%", you never take one.
The forecast isn't useless — **your rule is wrong**. Take an umbrella on the ten
wettest-looking days.

**Say it like this:**
> "I don't threshold at 0.5, I rank. The business rule is 'review the riskiest 10%',
> which depends only on the *order* of the scores, never their absolute size."

---

## 5. `class_weight='balanced'` changes alarm, not ranking — TESTED, REJECTED

| | flagged at 0.5 | mean probability | defaulters caught @10% |
|---|---|---|---|
| baseline | 2 | 3.69% | 1,678 |
| balanced | 64,304 | 44.64% | ~1,673 |

Enormous change in alarm level. **No change in who gets reviewed:** 91.4% of the
same people, Spearman rank correlation **0.9934**.

**Two doctors, same 100 patients.** One says 4% of you are at risk, the other says
45%. Ask each for their five most worrying patients — same five names.

**Say it like this:**
> "Class weighting essentially shifts the intercept. It moves every probability
> without reordering anyone. Since my decision rule uses rank, it bought nothing,
> so I kept the simpler model. If I want to flag more people I move the threshold —
> I don't reweight the model."

---

## 6. Random variation — the √n rule

A count moves around by chance even when nothing changed. Flip a coin 100 times and
you get 47 or 53, not always 50.

```
the wobble ≈ the square root of the count
  100 → ±10        1,678 → ±41        10,000 → ±100
```

It grows *slower* than the count, which is why large samples are more trustworthy.

**Say it like this:**
> "I sized my improvement threshold against the sampling noise. A count of about
> 1,700 carries roughly ±40 of random variation, so anything smaller than that I
> couldn't distinguish from a luckier validation split."

**Caveat, if pushed:** it's a rule of thumb. Comparing two models on the *same*
validation set is a paired comparison, so the true noise on the difference is
smaller than ±41. Erring conservative is the right direction for a decision bar.

---

## 7. Pre-registering the decision threshold

Before testing whether `credit_history_months` earned its place, the criterion was
**committed to git** (commit `5c060df`), then the result was committed separately
(`9b473f1`). Anyone can check the order.

**Say it like this:**
> "If you run the test first and then decide what counts as 'better enough', you'll
> look at a 0.7% improvement and talk yourself into whichever answer you already
> preferred. The only protection is committing to the threshold while you still
> don't know the result. Mine is in the git history, timestamped before the test."

---

## 8. Proxy discrimination — deleting one correlated feature is theatre

`credit_history_months` places a floor under a borrower's **age** across its whole
range. Age is protected under the **Equality Act 2010**.

But dropping it alone would have changed nothing: `mo_sin_old_rev_tl_op` correlates
with it at **0.92** and says the same thing under another name.

```
keep everything                          964 defaulters caught @5%
drop credit_history_months only          972      (information never left)
drop the 0.92 pair                       960
drop all age-linked                      961
```

All inside ±31. Removing them cost **nothing measurable**, so they went.

**Say it like this:**
> "The single-feature test showed no loss precisely *because* the information hadn't
> gone anywhere. If I'd stopped there I'd have written 'removed the age proxy at no
> cost' and it would have been false. Deleting one of two features correlated at
> 0.92 is theatre — you have to remove the cluster."

**Don't overclaim:** the model is **not** age-blind. `total_acc`, `num_rev_accts` and
`mort_acc` still correlate with age at 0.26–0.31 and were kept as legitimately
distinct measures. The honest claim is *"direct proxies removed at no measured cost;
residual correlation disclosed."*

**Also useful:** age is the one protected characteristic where different treatment can
be lawful if it's a proportionate means of achieving a legitimate aim. That defence
doesn't exist for race.

---

## 9. Outliers — signal or error?

A statistical outlier is not automatically bad data. In credit risk it's often the
customer you're trying to catch.

```
dti above 39     5,022 borrowers    6.41% default   ← strongest risk marker found
dti above 100       81 borrowers    1.23% default   ← BELOW base rate
dti = 999            5 borrowers    0.00% default
```

**The discriminator: risk climbs with dti, then collapses.** Real financial stress
doesn't reverse. That's where the data stops describing borrowers.

Percentile capping was **considered and rejected** — at 1st/99th it would have
altered ~10,000 rows per column and flattened every genuine signal. Instead 295
impossible values (0.03%) were set to **missing**: not deleted (that discards 60 good
columns over one bad field), not capped (that asserts a value you don't believe).

Effect: highest predicted probability fell from **100.0% to 56.3%**.

**Say it like this:**
> "One `dti` of 999 sat 101 standard deviations out and contributed about a hundred
> times what a normal borrower's did — it drowned the other 99 features. But I only
> cleared the impossible values, not the rare ones, because the rare ones were my
> best predictors."

**Errors cluster.** `annual_inc = 0` and `dti = 999` are the same defect — dti is
debt ÷ income, so a broken income field produces the absurd ratio. Likewise `pub_rec`
and `tax_liens`: 22 of 23 extreme rows are the same borrowers.

**What I'd do differently:** weight-of-evidence binning, which makes outlier treatment
largely irrelevant because you group values into bands rather than feeding raw
magnitudes to a linear model. Standard in scorecard development; out of scope here.

---

## 10. Coefficients mean "holding everything else equal"

`credit_history_months` came out **+0.056 — longer history, slightly MORE risk.**
Backwards from expectation.

**Why:** two borrowers both score 650. One has 25 years of credit history, the other
2 years. The 25-year borrower has had decades of chances and *still* only reached 650.
The credit score has already absorbed what history length would tell you; the leftover
meaning is "and they still only managed 650."

Same effect made `delinq_2yrs` (+0.051) look weaker than `fico_range_low` (−0.104) —
a FICO score is *built from* payment history, so it got there first.

**Say it like this:**
> "A small coefficient doesn't mean a useless feature — it often means another column
> already covers it. Drop FICO and `delinq_2yrs` would jump immediately. That matters
> if you're ever tempted to prune features by coefficient size."

---

## 11. Measuring an assumption instead of trusting it

The target needed "reached 90+ days past due within 12 months", but the data only
records one snapshot status per loan. It was reconstructed from the last payment date,
assuming a 3-month lag from last payment to 90 DPD.

That 3 was a guess — so it was **measured** against loans that were delinquent at the
snapshot:

```
paying normally        1 month since last payment
16-30 days late        2 months
31-120 days late       3 months (upper quarter: 4)
```

Each missed monthly payment moves you one step. The 90-day point sits at ~4 months,
not 3, so the cutoff was corrected from 9 to 8.

**Say it like this:**
> "Measuring the lag narrowed my default rate from 'somewhere between 4.38% and 6.51%'
> to '3.70%, at worst 4.38%' — about two-thirds less uncertainty. I assumed a 3-month
> lag, tested it, found 4, and corrected."

**The first row is the subtle bit:** even a perfect borrower is ~1 month past their
last payment, because payments are monthly. That gap is normal, not lateness — which
is why 90 days lands at 4 months rather than 3.

---

## 12. ROC-AUC, Gini, KS

**ROC-AUC 0.6976.** Pick one defaulter and one non-defaulter at random; how often does
the model score the defaulter higher? 70% of the time.
Floor is **0.50** (a coin flip), not 0. Below 0.50 means systematically wrong, which is
usable upside down.

**Gini 0.3952** = 2 × AUC − 1. Same information, rescaled so useless = 0.
**The credit-industry convention** — say "Gini 0.40" in a lending interview.

**KS 29.0.** Reading down the ranked list, the widest gap between how fast defaulters
accumulate and how fast non-defaulters do. At 34.4% down the list you've collected
62.3% of defaulters against 33.3% of non-defaulters. Typical range for an application
scorecard is 20–30.

**Is 0.70 good?** For application-only data, yes — published models sit around
0.65–0.72. You'd need behavioural data (months of payment history on *this* loan) to
beat it, and none of that exists on application day. **A model scoring 0.85 on this
dataset would almost certainly be leaking.**

**The KS catch, worth raising unprompted:**
> "My KS peaks 34% down the list, but I operate at 10%, where the gap is only 17.
> KS is a summary of separating power, not a guide to capacity — if I set my review
> volume from where a statistic peaked, I'd be letting a metric make a staffing
> decision."

---

## 13. The operating point is a business fact, not a model output

The model produces a **ranked list**. How far down you read is a staffing question.

```
review  1%  →  catch  3.8%    3.81x random
review  5%  →  catch 15.3%    3.07x
review 10%  →  catch 26.5%    2.65x   ← stated assumption
review 20%  →  catch 43.9%    2.20x
review 50%  →  catch 76.1%    1.52x
```

**The edge is biggest at the top** — 3.8x at 1%, down to 1.5x at 50%. Ranking is worth
most exactly when capacity is tightest. At 100% it's worth nothing, because you're
reviewing everyone either way.

**Say it like this:**
> "10% is an assumption, not a finding — it lives in one config constant with the full
> curve reported beside it. Give me a real capacity figure and every number updates.
> I report the curve rather than a single recall figure, because the trade-off is the
> useful part."

---

## 14. The honest limitations

Say these before you're asked.

- **At 10% capacity, 73.5% of defaulters are approved with no review.** The model
  reduces losses; it does not prevent them.
- **The target rests on a measured but unverifiable lag.** 3.70% at an 8-month cutoff,
  4.38% at 9 months.
- **Not age-blind.** Direct proxies removed; residual correlation of 0.26–0.31 remains
  in account-count features.
- **Reject inference not addressed.** The data contains only *approved* applicants, so
  the model is blind to everyone Lending Club already turned down.
- **US data, UK framing.** Lending Club is a US lender; the FCA and Equality Act
  discussion is educational, not a compliance claim.
- **No weight-of-evidence binning**, so large outliers in four retained columns still
  carry disproportionate leverage.

---

## 15. Things tried and rejected — with evidence

Worth more than a longer list of things used.

| Tried | Result | Decision |
|---|---|---|
| percentile capping (1st/99th) | would flatten `dti` 39-100, the strongest risk marker | rejected |
| `credit_history_months` | no measurable gain; proxies for age | dropped |
| `class_weight='balanced'` | 0.9934 rank correlation with baseline | rejected |
| SMOTE | *pending* | *pending* |

**Say it like this:**
> "I can tell you what I tried and didn't keep, and why, with numbers. Each of those
> was a decision with a threshold set in advance, not a preference."
