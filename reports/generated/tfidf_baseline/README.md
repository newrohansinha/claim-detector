# TF-IDF Logistic-Regression Baseline

All values below were generated from real predictions. The source-held-out models were
fit from scratch without any rows from the evaluation source.

| Evaluation | Samples | Accuracy | Claim precision | Claim recall | Claim F1 | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| mixed_paper_test | 2600 | 0.8465 | 0.8589 | 0.8030 | 0.8300 | 0.8451 |
| mixed_paper_test_without_train_duplicates | 2580 | 0.8453 | 0.8566 | 0.8000 | 0.8273 | 0.8436 |
| holdout_claimbuster | 7976 | 0.3699 | 0.2819 | 0.9829 | 0.4382 | 0.3604 |
| holdout_policlaim | 1953 | 0.6400 | 0.8855 | 0.4489 | 0.5957 | 0.6357 |
| holdout_averitec | 3067 | 0.5742 | 1.0000 | 0.5742 | 0.7295 | 0.3647 |
| external_checkthat | 911 | 0.6498 | 0.6557 | 0.9355 | 0.7710 | 0.5137 |

AVeriTeC contains only positive detection examples. Its apparent accuracy equals
recall, and its binary F1 is not compared as though it came from a two-class test.
