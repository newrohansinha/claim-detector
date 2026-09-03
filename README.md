# Claim Detector

This project investigates whether a sentence-level factual-claim detector that performs well on
a random mixed-source split still performs well when the evaluation source was never observed
during training.

The deployed task is **claim detection**, not fact verification: a false but externally
checkable assertion is still a claim.

## Hero investigation

The primary experiment compares ordinary mixed-source evaluation with source-held-out transfer.
The training corpus combines ClaimBuster, PoliClaim, and AVeriTeC, whose label distributions and
collection processes differ. The project tests, rather than assumes, whether those differences
permit source-specific shortcuts.

The investigation is pre-registered in
[`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md). Its label semantics are defined in
[`LABEL_POLICY.md`](LABEL_POLICY.md).

## Current status

The repository is under an explicit 20-hour timebox. Dataset acquisition, experiments, model
metrics, and serving artifacts must be generated from real upstream data and executable code.
No simulated metrics or dummy production model will be used.

The first verified data audit found 12,996 usable composite texts, one empty upstream AVeriTeC
claim, and 18 normalized sentence hashes crossing the paper's frozen train/test boundary. A
diagnostic that ignores sentence text and predicts only each source's majority label reaches
78.04% accuracy and 0.7389 claim F1 on the frozen test set. These facts motivate the hero
source-held-out experiment without yet proving that the language model uses source as a shortcut.

## Reproducibility entry points

```bash
make setup
make download
make prepare
make audit
make baseline
make verify
```

Raw third-party datasets and generated artifacts are deliberately excluded from Git. Acquisition
is pinned by upstream revision and SHA-256 checksum. See [`DATA_CARD.md`](DATA_CARD.md) for the
provenance and redistribution policy.
