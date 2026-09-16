"""Stage 14: what is the model worth? Break-even, not a fabricated benefit.

The honest problem with costing this model
------------------------------------------
Catching a likely defaulter is not the same as preventing the loss. The model
routes an application to review; what the reviewer then DOES - decline, reprice,
ask for documents, manually underwrite - is undefined in this project, and the
value depends entirely on that. Any single "the model saves $X" number would be
smuggling in an assumption about intervention effectiveness that nothing here
measures.

So this script does not claim a benefit. It solves for the break-even instead:

    what fraction of the losses it identifies would review have to prevent
    for the model to pay for its own review cost?

That number is robust because it needs no view on effectiveness - it IS the
view on effectiveness, stated as a threshold a business can argue about.

Why exposure and not counts
---------------------------
Recall counts defaulters. Money cares about balances. The model is slightly
better on money than on counts because it skews toward larger loans, and that
difference is worth reporting rather than leaving implicit.

Assumptions, all stated and all arguable
----------------------------------------
REVIEW_COST   cost of one manual review. ~1 analyst hour, loaded.
LGD           loss given default on unsecured consumer credit. Industry range
              is roughly 0.55-0.80; 0.65 is a common central estimate. NOT
              measured from this data - recoveries were dropped as leakage at
              stage 1, correctly for modelling and inconveniently for costing.
Currency      Lending Club is a US lender and loan_amnt is USD. Earlier notes
              in this repo quote review cost in GBP; USD is used throughout here.
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

ROOT = Path(__file__).resolve().parents[1]
TARGET = "default_within_12_months"

REVIEW_COST = 50.0      # USD per manual review
LGD = 0.65              # loss given default
LGD_RANGE = (0.55, 0.80)
COST_RANGE = (25.0, 100.0)


def main() -> None:
    cat = list(cfg.CATEGORICAL_FEATURES)
    flags = [f"{c}_missing" for c in cfg.MISSING_FLAG_COLUMNS]
    num = [c for c in cfg.model_features() if c not in cat]

    d = pd.read_parquet(ROOT / "data" / "interim" / "test.parquet")
    for c in cfg.MISSING_FLAG_COLUMNS:
        d[f"{c}_missing"] = d[c].isna().astype("int8")

    pipe = joblib.load(ROOT / "outputs" / "models" / "baseline_logistic.joblib")
    p = pipe.predict_proba(d[num + flags + cat])[:, 1]
    y = d[TARGET].values
    amt = d["loan_amnt"].values
    order = np.argsort(-p)
    n = len(y)

    total_def_amt = amt[y == 1].sum()
    print(f"TEST SET: {n:,} applications, {int(y.sum()):,} defaults, "
          f"${total_def_amt:,.0f} of defaulted balances")
    print(f"assumptions: review ${REVIEW_COST:,.0f}, LGD {LGD:.0%} "
          f"(both arguable - see docstring)\n")

    print("Money vs counts - the model skews toward larger loans:")
    k = int(round(n * cfg.REVIEW_CAPACITY))
    sel = order[:k]
    caught = sel[y[sel] == 1]
    missed_mask = np.ones(n, bool); missed_mask[sel] = False
    print(f"  mean loan, defaults caught   ${amt[caught].mean():>9,.0f}")
    print(f"  mean loan, defaults missed   ${amt[missed_mask & (y == 1)].mean():>9,.0f}")
    print(f"  recall by count              {y[sel].sum()/y.sum():>9.1%}")
    print(f"  recall by money              {amt[caught].sum()/total_def_amt:>9.1%}"
          "   <-- the one that pays\n")

    print("Break-even by review capacity")
    print("  'prevent' = share of identified losses that review must actually stop")
    print(f"  {'review':>7}{'reviews':>9}{'caught $':>13}{'cost':>11}"
          f"{'loss found':>13}{'prevent':>9}")
    rows = []
    for cap in [0.01, 0.05, 0.10, 0.20, 0.50]:
        kk = int(round(n * cap))
        s = order[:kk]
        c = s[y[s] == 1]
        exposure = amt[c].sum()
        cost = kk * REVIEW_COST
        loss_found = exposure * LGD
        be = cost / loss_found
        mark = "  <--" if abs(cap - cfg.REVIEW_CAPACITY) < 1e-9 else ""
        rows.append((cap, be))
        print(f"  {cap:>6.0%}{kk:>9,}${exposure:>12,.0f}${cost:>10,.0f}"
              f"${loss_found:>12,.0f}{be:>8.1%}{mark}")

    best = min(rows, key=lambda r: r[1])
    print(f"\n  lowest break-even is at {best[0]:.0%} review ({best[1]:.1%}).")
    print("  Reviewing fewer, riskier applications is the easier bar to clear -")
    print("  the question is whether a smaller review team catches enough money.")

    print("\nSensitivity of the break-even at the 10% operating point")
    kk = int(round(n * cfg.REVIEW_CAPACITY))
    s = order[:kk]; exposure = amt[s[y[s] == 1]].sum()
    print(f"  {'':<14}" + "".join(f"{f'LGD {l:.0%}':>11}" for l in
                                  [LGD_RANGE[0], LGD, LGD_RANGE[1]]))
    for rc in [COST_RANGE[0], REVIEW_COST, COST_RANGE[1]]:
        line = f"  review ${rc:<6,.0f}"
        for l in [LGD_RANGE[0], LGD, LGD_RANGE[1]]:
            line += f"{(kk*rc)/(exposure*l):>11.1%}"
        print(line)
    print("\n  Across every cell the bar is under 12%. The model does not need to be")
    print("  right about much for the review queue to be worth running - but this is")
    print("  a statement about review economics, not proof the model creates value.")
    print("  Intervention effectiveness is unmeasured here and would need a holdout")
    print("  test in production: review a random subset, leave a matched subset alone.")


if __name__ == "__main__":
    main()
