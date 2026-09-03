# Dataset Audit

This report is generated from checksum-verified upstream data. It contains measured
properties, not planned or simulated values.

## Source composition

| Source | Records | Positive | Negative | Positive rate | Median words | P95 words |
|---|---:|---:|---:|---:|---:|---:|
| averitec | 3067 | 3067 | 0 | 1.000 | 15.0 | 33.0 |
| claimbuster | 7976 | 1994 | 5982 | 0.250 | 15.0 | 46.0 |
| policlaim | 1953 | 1154 | 799 | 0.591 | 17.0 | 40.0 |

## Integrity findings

- Upstream composite records: 12997.
- Usable composite records: 12996.
- Invalid records: 1.
- Normalized duplicate groups: 49.
- Conflicting-label duplicate groups: 0.
- Cross-source duplicate groups: 0.
- Normalized hashes crossing the paper train/test boundary: 18.
- Records belonging to those cross-boundary groups: 54.

The frozen paper split is retained for comparability. Derived fit, validation, and
calibration partitions keep normalized duplicate groups together.

## Source-majority diagnostic

This diagnostic reads only source identity and predicts that source's training-set
majority label. It does not inspect sentence text and is not deployable.

| Metric | Value |
|---|---:|
| samples | 2600 |
| prevalence | 0.4665 |
| positive_prediction_rate | 0.3746 |
| accuracy | 0.7804 |
| claim_precision | 0.8296 |
| claim_recall | 0.6661 |
| claim_f1 | 0.7389 |
| macro_f1 | 0.7747 |
| true_negatives | 1221 |
| false_positives | 166 |
| false_negatives | 405 |
| true_positives | 808 |
| pr_auc | N/A |
| roc_auc | N/A |
