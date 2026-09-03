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

## Results

All numbers come from committed, text-free predictions over the real upstream data.

### Hero result: source-held-out transfer

Each row compares BERT on the same frozen paper-test records with and without that source available
during training. A fresh model was trained for every held-out condition. The held-out source was
also excluded from checkpoint selection.

| Evaluation source | Metric | Source included | Source held out | Change (paired 95% CI) | Actual claim rate | Held-out prediction rate |
|---|---|---:|---:|---:|---:|---:|
| ClaimBuster | Macro F1 | 0.8704 | 0.6737 | -0.1967 [-0.2184, -0.1752] | 0.2491 | 0.5400 |
| PoliClaim | Macro F1 | 0.8115 | 0.7243 | -0.0872 [-0.1272, -0.0472] | 0.5666 | 0.3107 |
| AVeriTeC | Claim recall | 0.9898 | 0.9628 | -0.0271 [-0.0406, -0.0152] | 1.0000 | 0.9628 |

Removing a source cuts macro F1 by 19.7 points on ClaimBuster and 8.7 points on PoliClaim. All
paired intervals exclude zero. The errors follow source-specific label priors: the ClaimBuster
holdout over-predicts claims, while the PoliClaim holdout under-predicts them. AVeriTeC is
positive-only, so recall is reported instead of binary F1.

### Benchmark checks

| Evaluation | Model | Accuracy | Claim F1 | Macro F1 | Predicted claim rate |
|---|---|---:|---:|---:|---:|
| Mixed test | TF-IDF | 0.8465 | 0.8300 | 0.8451 | 0.4362 |
| Mixed test | BERT | 0.9069 | 0.9007 | 0.9066 | 0.4712 |
| Mixed test without train duplicates | BERT | 0.9062 | 0.8993 | 0.9058 | 0.4678 |
| CheckThat tweets | TF-IDF | 0.6498 | 0.7710 | 0.5137 | 0.8990 |
| CheckThat tweets | BERT | 0.6400 | 0.7772 | 0.4200 | 0.9857 |

BERT reaches 0.9007 claim F1, compared with 0.911 in the
[paper](https://aclanthology.org/2025.fever-1.6.pdf). This run withheld 1,600 paper-training
records for validation and calibration instead of fitting on all of them.

The CheckThat result closely reproduces the paper: it reports 0.633 accuracy, 0.774 F1, and 0.998
recall; this run gets 0.640, 0.777, and 0.997. Macro F1 shows the failure hidden by claim F1: BERT
predicts `claim` for 98.6% of the tweets.

A source-only diagnostic reaches 0.7804 accuracy without reading the sentence. A grouped text
probe predicts source with 0.8613 accuracy. These are supporting diagnostics: they establish that
source signal exists, while the held-out experiment measures the resulting transfer gap.

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
make train-bert
make train-bert-heldout
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
- [`reports/generated/bert_mixed/metrics.json`](reports/generated/bert_mixed/metrics.json): training history, artifact hashes, and BERT metrics
- [`reports/generated/bert_heldout/metrics.json`](reports/generated/bert_heldout/metrics.json): source-held-out runs and matched transfer comparison
- [`reports/generated/source_probe/metrics.json`](reports/generated/source_probe/metrics.json): grouped source-probe result
