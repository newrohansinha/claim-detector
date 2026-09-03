# Pre-registered Experiment Protocol

## Hero question

Does a claim detector trained and evaluated on a random mixture of ClaimBuster, PoliClaim, and
AVeriTeC generalize to a collection source that was entirely absent during training?

## Hypotheses

1. **Source-held-out degradation:** claim-detection performance will be lower on an unseen source
   than on a mixed-source test set.
2. **Source recoverability:** a text-only source classifier will identify dataset provenance above
   the majority-source baseline.
3. **Source-label signal:** a source-majority diagnostic will attain nontrivial performance without
   reading sentence text.
4. **Calibration transfer:** temperature scaling will improve at least one in-domain calibration
   measure, but that improvement may not transfer to CheckThat tweets.
5. **Selective prediction:** removing the lowest-confidence predictions will reduce accepted-set
   error in-domain; out-of-domain behavior is an open empirical question.

These hypotheses are fixed before observing model results. Null, negative, and mixed results are
retained.

## Models

- Source-majority diagnostic using provenance metadata.
- Word unigram/bigram TF-IDF logistic regression.
- One fully fine-tuned `google-bert/bert-base-uncased` sequence classifier.

Multiple transformer families and broad hyperparameter searches are outside scope.

## Evaluation conditions

### Mixed source

Use the authors' frozen composite test set. Derive fit, validation, and calibration partitions only
from the released training portion, stratified over available source-label groups and grouped by
normalized text hash.

### Leave one source out

For each source, fit a fresh classifier using only the other sources and evaluate on the unseen
source. No examples from the held-out source may enter fitting, checkpoint selection, calibration,
or threshold selection.

AVeriTeC contains only positive detection examples, so its held-out evaluation reports recall,
false-negative rate, and score distributions rather than presenting binary F1 as comparable to
two-class test sets.

### External transfer

Evaluate the composite-trained model unchanged on CheckThat English tweets. CheckThat is not used
for base-model fitting or in-domain calibration.

### Human-reviewed challenge set

Evaluate a 60-example, single-annotator diagnostic set under the policy in `LABEL_POLICY.md`.
Examples remain excluded from training and model selection.

## Primary metrics

- Claim precision, recall, and F1.
- Macro F1 and accuracy.
- PR-AUC and ROC-AUC for two-class evaluations.
- Positive prediction rate and confusion matrix.
- Sample count and 95% bootstrap intervals.

Calibration analysis reports negative log loss, Brier score, Expected Calibration Error, and a
reliability diagram. Selective prediction reports risk versus automatic coverage.

## Interpretation constraints

- A source-held-out drop establishes a transfer problem, not its singular cause.
- A source probe establishes recoverable provenance signal, not that the claim model uses it.
- Temperature scaling is distribution-dependent and is not an OOD detector.
- A review flag is deployed only if its trigger identifies an elevated-error subset on held-out
  predictions.
- The test sets are not used to select model configuration or thresholds.

## Artifact requirements

Every reported table and figure must be regenerated from saved predictions and a versioned
configuration. Generated metrics record the model artifact hash, dataset manifest hash, Git commit,
and evaluation timestamp.

