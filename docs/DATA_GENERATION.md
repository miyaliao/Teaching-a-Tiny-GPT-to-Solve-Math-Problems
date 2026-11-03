# Data synthesis and historical Round9 Random100k

## What the selected training dataset actually was

The final Round9 Random condition used **100,000 positive/negative preference pairs**:

- 70,000 rows randomly selected from the original Round5 100k dataset as replay examples.
- 30,000 new rows sampled with replacement, with **2,500 rows per template across 12 math categories**.
- A common seeded permutation to interleave replay and new records.

The historical source implementation is [`scripts/build_round9_data.py`](../scripts/build_round9_data.py). That script uses `random.Random(42009)` for replay and new sampling, `420091` for the shared final shuffle, and seed `42` for canonical-fact train/validation/test partitioning.

The 12 categories are `add`, `subtract`, `multiply`, `divide`, `x_plus_a`, `a_plus_x`, `x_minus_a`, `a_minus_x`, `x_times_a`, `a_times_x`, `x_over_a`, and `a_over_x`. Chosen completions provide checked integer answers and explanations; rejected completions are an aligned refusal. New facts are checked against protected validation and teacher-benchmark facts. Historical replay rows have different provenance and can overlap with Teacher680.

## Distinguish two data generators

The generator embedded in [`dpo/dpo.ipynb`](../dpo/dpo.ipynb) creates a **standalone alternative** 100k-pair dataset using its own quotas and seeded sampling. It is **not** byte-for-byte identical to the archived historical Random100k.

The exact historical builder also needs older Round5 data, frozen teacher-style benchmarks and their manifests, and prior project utilities. Those input files are **not bundled** in this lightweight portfolio. Therefore the builder documents the original construction algorithm but cannot reproduce identical historical training bytes on its own.

The final data file `dpo/pos_neg_pairs.json` is in the separate course submission archive, not here. A bitwise historical reconstruction should first verify all original input file checksums.

## Evaluation caveat

Val768 and Teacher680 were development evaluations, not the TA private test. Teacher680 includes facts seen during historical training. It is therefore a diagnostic benchmark and does not establish unseen-fact generalization.
