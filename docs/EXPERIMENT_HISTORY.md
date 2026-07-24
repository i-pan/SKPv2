<!-- skpv2-project-documentation: experiment-history -->

# Experiment History

This is the project's experiment registry. Keep it current as experiments are
launched, stopped, fail, or become selected baselines.

This document is complementary to [DECISION_LOG.md](DECISION_LOG.md), and some
overlap is expected:

- **Experiment History** emphasizes what was run, its configuration, its result,
  and its disposition.
- **Decision Log** contains the deeper analysis explaining why an experiment or
  direction was selected, rejected, revised, or deferred.

Template-framework decisions belong in the repository-root
[`DECISIONS.md`](../DECISIONS.md), not here.

## Selected results

| Experiment | Configuration | Result | Decision |
|---|---|---|---|
| _Add the first baseline_ | _Model, data, split, resolution, schedule_ | _Primary metrics_ | _Selected, rejected, or pending_ |

## Active experiments

Record the run ID, config, folds, launch date, and current status for work that
is still running.

## Completed experiments

For each interpretable run, record:

- objective or hypothesis;
- exact config and run ID;
- data version and split contract;
- primary and important secondary metrics;
- comparison with the relevant baseline;
- artifacts or checkpoint location; and
- disposition and link to a detailed decision-log entry when appropriate.

## Failed or interrupted runs

Distinguish infrastructure and data-contract failures from models that trained
successfully but underperformed. A failed launch is not architecture evidence.

