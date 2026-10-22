---
title: Bit-exact predictions
description: What "equal to the library" means, how lignum guarantees it and how it is tested.
---

lignum's predictions are **identical** to the library's own `predict`: the same 64-bit values, not
values within a tolerance. The tests compare them with `np.testing.assert_array_equal`, never
`allclose`.

This matters in production: a model can be validated in a notebook with LightGBM or XGBoost and
deployed with lignum without a single prediction changing, so monitoring, thresholds and audits stay
valid.

## How

**The same floating-point operations, in the same order.** Floating-point addition is not
associative, so the sum of the trees depends on its order. lignum adds every tree's leaf to the
running sum in model order, exactly like the libraries, and never pre-sums groups of trees. Builds
use no `-ffast-math`, no FMA contraction (`-ffp-contract=off`, `/fp:precise`).

**Each library's split rules, translated exactly.** Parsers turn every split into one rule, *go right
if and only if `x >= threshold`, with NaN going a fixed direction*, choosing the threshold so the
answer is the library's for every possible input:

| Library | Native rule | In lignum |
|---|---|---|
| LightGBM | right if `x > t`; `\|x\| <= 1e-35` is treated as 0 | threshold `nextafter(t, +inf)`, moved to the edge of the zero band when `t` lies in it |
| LightGBM `missing_type=None` | NaN becomes 0.0 | NaN goes where 0.0 goes |
| LightGBM `missing_type=Zero` | NaN becomes 0.0, and 0 is missing | the zero band follows the missing direction |
| XGBoost | left if `float32(x) < t` (`t` is float32) | the smallest double that rounds to at least `t` in float32 |
| LightGBM categorical | NaN and negatives go right; `int(x)` in the set goes left | category sets per split |
| XGBoost categorical | `int(float32(x))` in the set goes right; invalid codes left | category sets per split |

**XGBoost's float32 arithmetic.** XGBoost accumulates leaves and applies transforms in float32, and
converts `base_score` to margin space in float32; lignum does the same. LightGBM uses double
throughout.

**Every kernel, the same answer.** The interpreter's two kernels, the compiled trees and every
thread layout produce identical results; the Python test suite runs every test three times (binned,
direct and compiled).

## Tested on hard inputs

The parity tests train real LightGBM and XGBoost models (every objective, multiclass, categorical
features, zero-as-missing, missing values off) and predict inputs that hit **every split threshold
exactly**, NaNs, zeros and values inside LightGBM's zero band, which is where precision mistakes
show up. They also run on five public data sets (California housing, Covertype, Adult, Bank
marketing, Diabetes 130-US hospitals, with up to 789 categories per feature).

Engines that are not exact differ in the last bits: in the [benchmarks](../../benchmarks/), lleaves
differed from LightGBM on 81% to 100% of the rows of the five real data sets.
