# Missing-data policy

Agreed 2026-09-07. Implemented in `src/config.py` and applied inside the modelling
pipeline so that every fill value is learned from the training pile alone.

## The principle

A blank cell is not one thing. Three different causes need three different responses,
and treating them identically would either destroy signal or invent it.

| Cause | Example in this data | Response |
|---|---|---|
| The blank *means* something | `mths_since_last_record` — 81.6% blank because most borrowers have no public record against them | Keep the fact of the blank as a flag |
| The field wasn't collected yet | The 14 Dec-2015 bureau fields | Drop the columns entirely |
| Genuinely absent, and absence is informative | `emp_length` — not stated | Make "Unknown" a category |

## What was dropped rather than filled

Fourteen credit-bureau columns were removed from the feature set before any
imputation was considered. They are missing for ~100% of 2015 originations and ~0%
of 2016 ones, because Lending Club only began collecting them in December 2015.
Their presence therefore encodes *when the loan was issued*, not anything about the
borrower.

Filling them, or adding missing-indicators for them, would have taught the model
"loans issued before December 2015 are safer" — a vintage artefact with no predictive
value for a new applicant, and one that would fail immediately in production.

See `docs/leakage_checklist.md` and `DROP_TIME_VARYING_AVAILABILITY` in `src/config.py`.

## The three rules

### Rule 1 — numeric columns: flag, then fill with the training median

For each numeric feature containing blanks, a binary companion column
`<name>_missing` is created, then the blank is filled with the median.

```
mths_since_last_delinq =    45     ->   45  ,  mths_since_last_delinq_missing = 0
mths_since_last_delinq = (blank)   ->   32  ,  mths_since_last_delinq_missing = 1
```

This separates two distinct questions that a single column cannot express:
*did the event ever happen?* (the flag) and *how long ago?* (the number).

**Why not a sentinel value such as 999?** Logistic regression fits a straight line
through the numeric range. Genuine values here sit around 5–80; inserting 999 would
exert enormous leverage on the fitted coefficient and distort the model. The flag
carries the same information without the distortion.

**Why the median rather than the mean?** Consumer-credit distributions are heavily
right-skewed — a small number of borrowers carry very large balances. The mean is
dragged by those; the median is not.

Flags are added where at least **0.5%** of training rows are blank (12 columns).
Below that threshold the flag would be estimated from too few rows to be meaningful,
so those columns (`dti`, `revol_util`, `inq_last_6mths`, `num_rev_accts`) receive the
median fill alone.

### Rule 2 — `emp_length`: blank becomes its own category

`emp_length` is text, and its blank is informative:

| | Default rate |
|---|---|
| `emp_length` missing | 4.92% |
| `emp_length` present | 3.62% |

Borrowers who do not state employment length default about 1.3 pp more often.
Rather than imputing a length, "Unknown" is treated as a legitimate category.

### Rule 3 — every fill value is learned from the training pile only

Medians are computed on the 513,300 training rows and applied unchanged to
validation and test. Computing them across the full dataset would let information
from the sealed test set influence training — the same leakage principle that governs
the column policy, applied to preprocessing.

This is also why the train/validation/test split happens *before* any imputation
(`src/03_split.py`), not after.

## Effect on the feature set

| | Count |
|---|---|
| Features after the column policy | 60 |
| Missing-indicator flags added | 11 |
| Columns entering the model | 71 |

(Before one-hot encoding of the five categorical features.)

## Known limitation

The median fill is unconditional: every borrower with a blank
`mths_since_last_delinq` receives the same value. A more sophisticated treatment
would model the missing value from the borrower's other attributes. That was not
done here, and the flag columns are what carry the information instead.
