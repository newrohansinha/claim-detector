# Text-to-Source Probe

This is a five-fold stratified group cross-validation experiment. Normalized duplicate texts stay
within one fold. It measures whether source provenance is recoverable from sentence text; it does
not prove that the claim classifier uses source information.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Majority source | 0.6137 | 0.2535 |
| TF-IDF source probe | 0.8613 | 0.7848 |
