"""Stage 4: stratified train / validation / test split.

Three piles:
    train (60%)  - the model learns from these
    val   (20%)  - used to compare settings and pick a model
    test  (20%)  - sealed until the very end, opened once

`stratify=y` forces every pile to carry the same share of defaulters as the
full dataset. With 855k rows a plain random split would land close anyway, but
stratifying makes it exact rather than lucky - and the habit matters when the
positive class is rare, which here it is (3.70%, a 26:1 imbalance).

The split happens BEFORE any cleaning, imputation or encoding. That ordering is
deliberate: filling missing values using statistics computed over the whole
dataset would leak information from test into train.

Nothing is duplicated or discarded - every row lands in exactly one pile.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
SRC = INTERIM / "cohort_2015_2016.parquet"

TARGET = "default_within_12_months"
TEST_SIZE = 0.20
VAL_SIZE = 0.20        # of the whole, i.e. 0.25 of the post-test remainder
RANDOM_STATE = 42


def describe(name: str, y: pd.Series, total: int) -> None:
    n, k = len(y), int(y.sum())
    print(f"  {name:<8}{n:>9,}  ({n / total:>5.1%} of data)   "
          f"defaults {k:>7,}   rate {k / n:.4%}")


def main() -> None:
    df = pd.read_parquet(SRC)
    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    print(f"Loaded {SRC.name}: {len(df):,} rows x {X.shape[1]} features")
    print(f"Overall default rate: {y.mean():.4%}\n")

    # First cut: hold out the test pile.
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Second cut: split the remainder into train and validation.
    # VAL_SIZE is a share of the WHOLE, so rescale it against what is left.
    val_share_of_rest = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_share_of_rest,
        stratify=y_rest, random_state=RANDOM_STATE
    )

    total = len(df)
    print("Split result:")
    describe("train", y_train, total)
    describe("val", y_val, total)
    describe("test", y_test, total)

    # --- Checks -----------------------------------------------------------
    assert len(X_train) + len(X_val) + len(X_test) == total, "rows lost or duplicated"
    ids = set(X_train.index) | set(X_val.index) | set(X_test.index)
    assert len(ids) == total, "a row appears in more than one pile"

    rates = [y_train.mean(), y_val.mean(), y_test.mean()]
    spread = (max(rates) - min(rates)) * 100
    print(f"\n  every row accounted for exactly once: {total:,}")
    print(f"  default rate spread across piles: {spread:.4f} pp")

    for name, X_part, y_part in [("train", X_train, y_train),
                                 ("val", X_val, y_val),
                                 ("test", X_test, y_test)]:
        part = X_part.copy()
        part[TARGET] = y_part
        out = INTERIM / f"{name}.parquet"
        part.to_parquet(out, index=False)
        print(f"  saved {out.relative_to(ROOT)}  ({len(part):,} rows)")


if __name__ == "__main__":
    main()
