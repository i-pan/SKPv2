# SKPv2 Agent Contract

These instructions apply to agents working in the SKPv2 template or in a
project cloned from it.

## Determine the repository role first

Before editing anything, inspect `git remote -v`, the current branch,
`git status --short`, and the top-level files.

- If `origin` is the SKPv2 repository, this is the reusable template. Keep
  changes generic and record template/framework decisions in root
  `DECISIONS.md`.
- If a remote named `template` points to SKPv2 and `origin` points elsewhere,
  this is a downstream experiment project. Keep project-specific work here and
  record its decisions in `docs/DECISION_LOG.md`.
- Never assume a clean worktree. Existing changes belong to the user unless
  proven otherwise.

## Bootstrap a new project

When the user provides the SKPv2 GitHub URL and a new project-repository URL
from an empty working directory, perform the setup directly:

```bash
git clone <skpv2-template-url> .
git remote rename origin template
git remote add origin <new-project-repo-url>
uv sync --group train --group dev
```

If the directory is not empty, inspect it and preserve its contents. Do not
delete or replace existing files without explicit authorization. Verify the
remote layout after setup:

```text
origin    <new project repository>
template  <SKPv2 template repository>
```

Do not push merely because setup completed. Push only when requested.

## Mandatory orientation before project work

Before designing the first experiment, build a comprehensive understanding of
the clone. Read and inspect:

1. `README.md`, `QUICK_START.md`, this file, and root `DECISIONS.md`;
2. `docs/EXPERIMENT_HISTORY.md` and `docs/DECISION_LOG.md`;
3. `pyproject.toml`, `.env.example`, and the resolved dependency groups;
4. config defaults and starter configs under `src/skp/configs/`;
5. the training entry point, runners, tasks, samplers, callbacks, and logging;
6. relevant dataset, model, loss, and metric contracts;
7. tests that exercise the intended extension points; and
8. repository status, remotes, available hardware, and configured data paths.

Use `rg` and `rg --files` for discovery. Read complete instruction and contract
files rather than relying on filenames or partial snippets.

Then inspect the project data before modeling:

- enumerate files and tables;
- measure row counts, missingness, label prevalence, and annotation coverage;
- identify patient/study grouping and leakage risks;
- inspect image dimensions, channels, and representative samples;
- confirm the intended fold and validation contract; and
- document assumptions and unresolved risks.

Report the understood architecture, extension points, data contract, validation
strategy, and any genuine blockers before launching expensive training.

## Required project documentation

Every clone must retain these distinct records:

- Root `DECISIONS.md` is the inherited SKPv2 template/framework history.
- `docs/EXPERIMENT_HISTORY.md` tracks project experiments, configurations,
  results, failures, and dispositions.
- `docs/DECISION_LOG.md` contains detailed project-specific reasoning,
  evidence, alternatives, tradeoffs, and downstream consequences.

Experiment History and Decision Log are complementary and may intentionally
overlap. Keep their `skpv2-project-documentation` marker comments so the
documentation-contract tests continue to pass.

Do not put project modeling, dataset, evaluation, or release decisions into root
`DECISIONS.md`. When a project reveals a reusable framework improvement, port it
to the template deliberately and document that separate template decision
there.

## MLflow is fail-closed

Normal training requires a configured, reachable non-local MLflow tracking
backend. `skp-train` performs a preflight before it creates the experiment run
directory or initializes training.

- If `MLFLOW_TRACKING_URI` is missing, local, or unreachable, do not launch.
- Tell the user that the MLflow backend must be started or corrected.
- Never add `--allow_local_mlflow` on your own.
- Use `--allow_local_mlflow` only when the user explicitly authorizes local
  tracking for that launch. The authorization is per launch and should not be
  made a persistent config default.
- `--dry_run` is an explicitly local, temporary diagnostic mode and does not
  authorize a real training run.

Verify and report the credential-redacted tracking URI before long-running
training. On this workstation the standard server endpoint in `.env.example` is
`http://127.0.0.1:5001`; a SQLite backend URI does not prove that server is
running and is therefore treated as local. Do not expose MLflow credentials in
logs or documentation.

## Building experiments

- Copy and specialize the nearest config/template instead of introducing deep
  inheritance without need.
- Preserve strict config access and explicit optional defaults.
- Keep reusable framework logic separate from project-specific data and label
  contracts.
- Make split columns, fold routing, target ordering, normalization, resize
  behavior, loss masking, and validation metrics explicit.
- Validate a new pipeline with focused tests, a dataset sample, model
  construction, checkpoint loading when relevant, and a short smoke run before
  launching a full queue.
- Use BF16 where supported unless the experiment requires another precision.
- Run long jobs in persistent sessions, mirror console logs into each experiment
  directory, and monitor MLflow plus system/GPU metrics.
- Never interpret sanity-check metrics from only a few batches as model
  performance.
- Distinguish code/data-contract failures from trained but underperforming
  models in Experiment History.

## Completion standard

Before calling setup or implementation complete:

- run relevant focused tests and then the full test suite when practical;
- run linting and `git diff --check`;
- verify documentation was updated at the appropriate level;
- summarize files changed, tests run, active processes, and remaining risks; and
- state explicitly whether anything was committed, pushed, deployed, or launched.
