# Timebox

The project is limited to 20 hours of active work. Model training elapsed time, substantial Codex
work, validation, and documentation are recorded rather than hidden.

## Work log

| Date | Phase | Activities | Status |
|---|---|---|---|
| 2026-09-02 | Foundation | Repository, research contract, label policy, reproducible environment | Complete |
| 2026-09-02 | Data | Pinned acquisition, exact split reconstruction, integrity tests, generated audit | Complete |
| 2026-09-02 | Baselines | TF-IDF mixed/held-out transfer runs, bootstrap intervals, grouped source probe | Complete |
| 2026-09-02 | Fine-tuning | Pinned BERT mixed, CheckThat, and three source-held-out runs | Complete |
| 2026-09-02 | Verification | Linux CI with real pinned data acquisition, tests, and package build | Complete |
| 2026-09-03 | Controlled evaluation | Size/class-prior-matched BERT controls and paired intervals | Complete |
| 2026-09-03 | Confidence | Reserved-split temperature scaling and review-policy transfer test | Complete |
| 2026-09-03 | Target adaptation | Repeated disjoint threshold selection on real CheckThat predictions | Complete |
| 2026-09-03 | Serving | Validated FastAPI contract, hardened Docker image, real HTTP load measurement | Complete |

## Scope rule

Committed components are completed against their acceptance criteria. Additional techniques are
added only in response to measured evidence; no placeholders or simulated results are used to
claim completion.
