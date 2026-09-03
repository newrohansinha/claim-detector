# Claim Detector

[![CI](https://github.com/newrohansinha/claim-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/newrohansinha/claim-detector/actions/workflows/ci.yml)

Detect whether an English sentence contains a factual claim. This is not fact verification: a
false but checkable statement is still a claim.

## Main question

Does strong performance on a random mix of ClaimBuster, PoliClaim, and AVeriTeC survive when the
test source was never seen during training?

That source-held-out comparison is the main experiment. Dataset source and label are strongly
related, so a random split may reward source-specific patterns. The experiment tests that
hypothesis; it does not assume it is true.

## Results so far

All numbers below come from saved predictions over the real upstream data.

| Experiment | Accuracy | Claim F1 | Macro F1 | Predicted claim rate |
|---|---:|---:|---:|---:|
| Source-only diagnostic, mixed test | 0.7804 | 0.7389 | 0.7747 | 0.3746 |
| TF-IDF, mixed test | 0.8465 | 0.8300 | 0.8451 | 0.4362 |
| TF-IDF, mixed test without train duplicates | 0.8453 | 0.8273 | 0.8436 | 0.4326 |
| TF-IDF, ClaimBuster held out | 0.3699 | 0.4382 | 0.3604 | 0.8716 |
| TF-IDF, PoliClaim held out | 0.6400 | 0.5957 | 0.6357 | 0.2995 |
| TF-IDF, CheckThat tweets | 0.6498 | 0.7710 | 0.5137 | 0.8990 |

The source-only diagnostic does not read sentence text. It predicts each source's majority
training label. Its performance shows why source-aware evaluation matters.

A grouped five-fold TF-IDF probe predicts the source dataset from sentence text with 0.8613
accuracy and 0.7848 macro F1. This shows that source is recoverable from text. It does not prove
that the claim classifier uses source as a shortcut.

The TF-IDF result drops sharply when a source is unseen. CheckThat claim F1 looks reasonable, but
the model predicts `claim` for 89.9% of tweets; macro F1 exposes that failure.

The transformer experiment is next. No conclusion about BERT transfer is claimed yet.

## Data checks

- 12,997 upstream composite rows; 12,996 contain usable text.
- One AVeriTeC row has an empty claim.
- 18 normalized sentence hashes cross the paper's frozen train/test boundary.
- New fit, validation, and calibration splits keep duplicate groups together.
- Raw third-party data is downloaded at a pinned revision and verified by SHA-256.

The frozen split is retained for comparison with the paper. A second result removes test records
whose normalized text appears in training.

## Reproduce

Requirements: `uv` and Python 3.12.

```bash
make setup
make prepare
make audit
make baseline
make source-probe
make verify
```

Generated evidence is in [`reports/generated`](reports/generated). Raw data and model binaries are
not committed. The downloader reconstructs them from pinned sources.

## Files worth reading

- [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md): hypotheses and evaluation rules
- [`LABEL_POLICY.md`](LABEL_POLICY.md): what counts as a claim
- [`DATA_CARD.md`](DATA_CARD.md): source composition, licenses, and integrity findings
- [`reports/generated/data_audit/data_audit.md`](reports/generated/data_audit/data_audit.md): measured data audit
- [`reports/generated/tfidf_baseline/metrics.json`](reports/generated/tfidf_baseline/metrics.json): full baseline metrics and bootstrap intervals
- [`reports/generated/source_probe/metrics.json`](reports/generated/source_probe/metrics.json): grouped source-probe result
