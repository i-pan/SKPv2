<!-- skpv2-project-documentation: decision-log -->

# Project Decision Log

This file records detailed, project-specific decisions and analysis. Each
material decision should capture its context, evidence, alternatives,
tradeoffs, conclusion, and consequences.

> **This file is not `DECISIONS.md`.**
>
> The repository-root [`DECISIONS.md`](../DECISIONS.md) records decisions about
> developing and maintaining the reusable **SKPv2 template/framework**.
> `docs/DECISION_LOG.md` records decisions about the **experiment or product
> being built from that template**. Do not put modeling, dataset, evaluation, or
> release decisions for the current project into the template decision history.

This document is complementary to
[EXPERIMENT_HISTORY.md](EXPERIMENT_HISTORY.md), and intentional overlap is
expected:

- **Experiment History** is primarily an experiment tracker: what ran, how it
  was configured, what happened, and whether it was retained.
- **Decision Log** provides the more detailed reasoning and analysis: why the
  experiment was run, how evidence was interpreted, which alternatives were
  considered, and what the result changes downstream.

An important experiment will often have a concise row or section in Experiment
History and a fuller entry here.

## Decision-entry template

### YYYY-MM-DD: Decision title

#### Context

What problem or uncertainty prompted the decision?

#### Evidence

What measurements, qualitative review, failure analysis, or external references
informed it? Clearly distinguish preliminary, cross-validation, holdout, and
external-test evidence.

#### Alternatives considered

What other viable options were considered, and what were their tradeoffs?

#### Decision

What was selected, rejected, deferred, or left conditional?

#### Consequences and follow-up

What changes in the training, evaluation, inference, documentation, or release
contract? What still needs to be tested?

