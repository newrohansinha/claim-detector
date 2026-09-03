# Data Card

## Intended use

The data supports controlled experiments in English sentence-level claim detection, with special
attention to transfer across collection sources. It is not a truthfulness or evidence-verification
dataset.

## Sources

The experiment follows the composite construction released with Bell (2025): ClaimBuster,
PoliClaim, and AVeriTeC for training and in-domain evaluation, plus CheckThat 2022 English tweets
for external transfer evaluation.

Acquisition is pinned to VeritaResearch/claim-extraction revision
`4fd0cbe0f74fb08d3caf76d77f6757fc9207ebe9`. Individual files are verified against the checksums
in `src/claim_detector/data/sources.json`.

## Known structural differences

- ClaimBuster contributes 7,976 binary-labeled sentences in the released artifact.
- PoliClaim contributes 1,953 sentences from political speech and debate sources.
- AVeriTeC contributes 3,068 claims and no matched negative examples to the detection dataset.
- CheckThat contains 911 English tweets and is reserved for external transfer evaluation.

AVeriTeC's single-class slice cannot support ordinary binary precision/F1 interpretation on its
own. Evaluation on that source will emphasize recall, false-negative rate, and predicted
probability distribution.

## Redistribution

The upstream research repository does not currently provide a repository-level license file.
This project therefore does not commit or redistribute the upstream data. The acquisition command
downloads the pinned public research artifacts for local reproducibility and records checksums.
Each original dataset's license and terms must be reviewed before any separate redistribution or
commercial use.

## Processing commitments

- Preserve source provenance on every record.
- Preserve original text alongside a normalized hash used only for integrity checks.
- Reject empty text and invalid labels.
- Detect identical text with conflicting labels.
- Keep normalized duplicate groups from crossing derived development splits.
- Never tune model configuration or confidence thresholds on the final test data.

