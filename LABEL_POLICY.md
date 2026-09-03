# Claim Label Policy

## Operational definition

A sentence contains a factual claim when it presents at least one proposition about the world
whose truth could, in principle, be assessed using external evidence.

This policy concerns whether a claim is present. It does not ask whether the claim is true,
important, persuasive, or supported by available evidence.

## Positive cases

- True and false factual assertions.
- Negated factual assertions.
- Quantitative assertions.
- Attributed assertions, including assertions about what a person or organization said.
- Definite predictions that will become externally testable.
- Mixed sentences containing at least one checkable proposition.

## Negative cases

- Pure questions that do not themselves assert a proposition.
- Commands and requests.
- Greetings and conversational acknowledgements.
- Pure preferences or value judgments.
- Emotional expressions without a factual proposition.
- Fragments that do not assert anything.

## Hard cases

| Sentence | Intended label | Rationale |
|---|---:|---|
| The Empire State Building is the tallest building in New York City. | Claim | It is externally checkable even if false. |
| Taylor Swift is the greatest singer alive. | Not claim | "Greatest" has no agreed factual criterion in this context. |
| John said the company lost five million dollars. | Claim | The attribution and reported content are checkable. |
| Is it true that unemployment increased last year? | Not claim | Under the sentence-level policy, this asks rather than asserts. |
| Tomorrow it will rain in Boston. | Claim | It is a definite prediction that will become checkable. |
| If taxes rise, unemployment might increase. | Not claim | The hedged hypothetical lacks a definite asserted outcome. |

## Dataset-policy warning

The source datasets were built for related but potentially different targets, including factual
claim detection and check-worthiness. Their labels are retained for paper-comparable experiments,
but they are not automatically treated as perfect implementations of this policy.

Cross-source disagreement can therefore reflect both language shift and annotation-policy shift.
The final report must not attribute every transfer error to writing style.

## Challenge-set governance

The challenge set will not be used for training. Candidate examples may be proposed with Codex,
but Rohan Sinha must approve every sentence, label, and rationale before evaluation. The set will
be described as a small, single-annotator diagnostic rather than a representative benchmark.

