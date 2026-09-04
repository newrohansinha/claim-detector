# Experiment Protocol and Analysis Status

## Main question

When a claim detector reaches a new domain and begins labeling almost everything as a claim, has
it lost the ability to separate the classes, or has its decision threshold stopped transferring?

The distinction is operationally important. Representation failure requires new modeling or
training data. Threshold failure may be recoverable with a small labeled target sample while the
model weights remain frozen.

## Analysis status

The mixed and CheckThat evaluations are frozen benchmark reproductions. The target-threshold
adaptation study is a post-hoc exploratory analysis prompted by CheckThat's 98.6% positive
prediction rate and BERT's simultaneously nontrivial ROC-AUC. It is not presented as a
preregistered or untouched-target result.

The adaptation budgets, split rule, selection metric, repetitions, seed, and reporting interval
were fixed before the recorded 2,000-repeat run. A smaller development pilot was used only to
decide whether this follow-up question was worth pursuing.

The earlier source-transfer hypotheses and matched source-exposure control remain documented
below as supporting analyses. They are not the project's primary claim.

## Models and fixed evaluation artifacts

- A full fine-tune of `google-bert/bert-base-uncased`.
- A word unigram/bigram TF-IDF logistic-regression baseline.
- The 911 CheckThat English COVID-19 tweets are external-only target data.
- Both models are compared on the same aligned CheckThat records.
- BERT probabilities receive temperature scaling fit only on 800 reserved composite records.
- No model weights are updated in the target-threshold experiment.

The threshold study consumes saved, text-free prediction artifacts. It does not regenerate,
rewrite, augment, or synthesize evaluation sentences.

## Target-threshold adaptation

### Diagnostic stage

Evaluate each frozen model on all CheckThat records at the fixed 0.5 threshold. Report:

- claim F1 and macro F1;
- claim and non-claim recall;
- positive prediction rate and confusion counts; and
- ROC-AUC and PR-AUC.

ROC-AUC is threshold-free and tests whether class ranking remains useful. An oracle threshold is
also computed on all CheckThat labels as an explicit upper-bound diagnostic. Because it is chosen
and evaluated on the same labels, it is never reported as deployable performance.

### Adaptation stage

For labeled target budgets of 10, 25, 50, 100, and 200:

1. Draw a stratified adaptation subset from CheckThat.
2. Select the probability threshold that maximizes macro F1 on that subset.
3. Break exact ties with the median maximizing threshold.
4. Evaluate the threshold on the disjoint remaining records.
5. Use the identical split for BERT and TF-IDF.
6. Repeat 2,000 times with seed 42 plus the adaptation budget.

Report the median, mean, and central 95% range across repeated splits for macro F1, improvement
over the fixed threshold on the same evaluation records, both class recalls, positive prediction
rate, and selected threshold.

The 2.5th–97.5th percentile range describes variation across labeled adaptation samples. It is not
a population confidence interval, and the repeated evaluations are not independent datasets.

## Supporting source analyses

The original investigation tested whether random mixing across ClaimBuster, PoliClaim, and
AVeriTeC concealed dependence on dataset source.

### Source-held-out evaluation

For each source, fit a fresh BERT classifier on the other sources and evaluate on the absent
source. No examples from the target source enter fitting or model selection. AVeriTeC contains
only positive examples in this release, so its source slice is evaluated with claim recall.

### Matched source-exposure control

For each source-held-out model, train a source-included control with exactly the same fit and
validation row counts and label counts. Preserve normalized-text groups, require the target
source in fitting and validation, and exclude hashes from that source's frozen test records.
Compare aligned predictions with a paired 2,000-sample bootstrap.

This control removes training-size and class-prior differences. Restoring the target source still
changes the remaining source mixture, so the result is evidence of source dependence rather than
a singular causal mechanism.

### Source diagnostics

- Predict claim labels from source identity without sentence text.
- Predict source identity from text in grouped five-fold cross-validation.
- Compare with the majority-source baseline.

These establish recoverable provenance signal; they do not prove that BERT uses a specific source
shortcut.

## Interpretation constraints

- Similar claim F1 does not imply similar two-class behavior when prevalence and prediction rate
  differ.
- ROC-AUC establishes ranking quality, not a usable deployed operating point.
- Target-threshold adaptation assumes access to representative labeled target examples.
- Optimizing macro F1 encodes equal importance for both classes; a production cost function may
  require another threshold.
- Temperature scaling learned in-domain is not an out-of-domain safeguard.
- The target adaptation result is exploratory and limited to one external dataset.
- The source-transfer analyses establish domain dependence, not its single cause.

## Artifact requirements

Every reported number and figure must be reproducible from versioned code and saved prediction
artifacts. Generated evaluation files contain record identifiers, labels, hashes, probabilities,
and predictions, but no sentence text. Metrics record the relevant artifact hashes, split rules,
seeds, and analysis status.
