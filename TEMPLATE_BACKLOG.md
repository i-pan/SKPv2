# Template Backlog

Use this file to collect small reusable improvements discovered while working on
experiment projects. The goal is to avoid pushing tiny template commits every
time a minor bug, cleanup, or ergonomic tweak appears.

## How To Use

Add short entries under `Open Items` when a change seems broadly useful for
future SKPv2 projects but is not urgent enough to patch immediately.

Prefer immediate commits for:

- Bugs that can corrupt training, evaluation, checkpoints, or saved artifacts.
- Security, dependency, or environment fixes.
- Documentation errors that would block a new user from getting started.
- Changes needed by multiple active projects right now.

Batch later for:

- Small naming/docstring cleanups.
- Convenience defaults.
- Minor template polish.
- Ideas that need another real experiment before they prove their shape.

When batching items into a template update, move completed bullets to
`Completed Items` with the commit hash or date.

## Open Items

- **A04 — hypothesis requiring verification:** GPU/NCCL, mixed precision,
  SyncBatchNorm and chunked-encoder collective integration. CPU/gloo tests pass;
  run an explicitly authorized bounded GPU check before broadening support.
- **O04 — optional extension:** CSN/R(2+1)D and Swin checkpointing require
  architecture/shape and pretrained-weight coverage before migration.
- **E01 — reusable improvement:** Extend exact validation beyond map-style
  classification only after task-specific empty-rank and collective tests.
- **W04 — optional extension:** Larger prediction corpora may need streaming
  exports; current opt-in gather is deliberately for modest tables.

## Completed Items

2026-11-08, uncommitted reviewable implementation (verification in migration record):

- **R07–R13 — deep-review integration corrections:** weights-only EMA saving,
  skipped optimizer updates, built-in and combined loss reductions, fractional
  curve timing, unset CLI overrides and checkpoint symlink boundaries. Final
  affected contracts independently cross-reviewed; CPU/gloo regressions retained.
- **A05 — reusable review workflow:** repository-local deep-review skill included
  in the prepared commit, with contract, adversarial and final verification passes.

- **R01–R06 — confirmed integration bugs/guardrails:** inference versus resume
  checkpoint loading, tied EMA state, finite metric inputs, distributed export
  failure propagation, explicit OneCycle horizons and initialized topology checks.

- **S01–S05 — confirmed bugs/reusable checkpoint improvement:** sampler-aware
  schedule sizing/dispatch, recursive config copy, optimizer-step EMA, linear
  default ramp, raw/inference checkpoint separation and saved resume settings.
- **E01–E04 — evaluation:** exact-once opt-in classification (including empty
  rank control batches), numerator/denominator loss aggregation, mapped-target
  validity counts, atomic prediction export and exact ID coverage checks.
- **W01–W04 — workflow:** opt-in environment profile, seed/topology controls,
  generic curves, immutable manifests/source snapshots/registry scaffolding and
  optional prompt/result bindings.
- **O01–O03 — reusable optional components:** deterministic group sampling,
  masked soft BCE, MIL feature pooling/chunk controls and DICOM header/serial
  utilities. CPU acceptance is bounded; GPU and real corpus checks remain open.
- **A01 — review skill:** audit-only skill, references, bounded standard-library
  inventory helper and classified machine-readable findings.

## Retained classifications

- **A02 — already upstream:** final checkpoint callback, console capture,
  MLflow/documentation contracts, normalized-LSE baseline MIL and optional groups.
- **A03 — project-only:** competition labels/cohorts, clinical routing, private
  datasets/paths, submission queues/fallbacks and historical experiment registries.

See [migration contracts](docs/UPSTREAM_MIGRATION.md) and
[machine-readable findings](docs/UPSTREAM_FINDINGS.json) for evidence and limits.
