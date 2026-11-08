---
name: experiment-cleanup
description: Audit experiment storage, results, reproducibility and deployment dependencies to propose checkpoint cleanup. Use when experiments accumulate or storage needs reclaiming. Never remove or alter experiment artifacts without explicit approval of a concrete cleanup proposal.
---

# Experiment cleanup

Recover storage primarily from obsolete checkpoints while preserving useful
results and enough configuration evidence to reproduce their experiments approximately.
Exact historical code and bitwise-identical weights are not required by default. The default output
is a proposal, never cleanup. Age or a low score alone does not establish that a
checkpoint is disposable.

## Authority and boundaries

- NEVER delete, prune, truncate, overwrite, move, compress, offload or otherwise
  change experiment artifacts without the user's explicit approval of the
  concrete actions and artifacts. This includes tracking-service artifacts,
  caches, remote objects, optimizer-state stripping and entire run directories.
- A request to audit, propose, save space, or use this skill is not deletion
  approval. Previous cleanup approval does not authorize a new proposal or
  newly discovered files. Silence, time elapsed, disk pressure and an automated
  approval are not user approval. Do not reinterpret ambiguous assent as approval.
- Read-only discovery and writing a new proposal/reproduction record in a safe
  documentation location are allowed. Do not overwrite prior proposals or
  expose credentials, private samples or report text in evidence. Do not modify
  live registries while auditing. Never launch training, inference, downloads,
  migrations or cloud spending to establish whether something can be deleted.
- Preserve existing dirty work. Determine repository role from instructions,
  remotes, branch and status. Keep generic skill decisions in template
  `DECISIONS.md`; keep actual project cleanup decisions in its decision log.
  Do not run a cleanup audit merely because asked to create or edit this skill.

## Discover the evidence and storage

Read [the proposal contract](references/proposal.md) when preparing an audit.
Find actual experiment roots and artifact backends from project instructions,
configs, tracking configuration and launch paths. Inspect histories, decision
logs, experiment/dataset/prompt registries, run manifests, logs, saved configs,
metric histories and inference packaging. Do not import arbitrary configs or
unpickle checkpoints to inspect them; prefer text, existing manifests and file
metadata. Bound individual reads and paginate large registries without silently
limiting the audit to the most recent or successful runs.

Inventory all discovered experiments, including failed/interrupted/abandoned
runs and files absent from registries. Reconcile registered runs with local and
remote artifacts. State inaccessible backends, missing history, stale records,
and exclusions; do not call a partial inventory complete. Use existing read-only
filesystem/backend tools; no destructive helper or cleanup command is needed
for the proposal. Start with metadata sizing, not a full read/hash of every
multi-gigabyte checkpoint. Record hashes where available and unknowns explicitly.

Measure logical and allocated bytes where available. Resolve symlinks without
following them into unapproved scan roots; identify hardlinks and shared storage
so aliases do not inflate recoverable space. Treat remote replicas, snapshots,
reflinks, sparse files and open deleted files as limits on reclamation estimates.
Do not assume `.pt`, `.pth` or `.ckpt` identifies a disposable checkpoint. A large
file's role must be established; datasets and pretrained inputs are outside the
default cleanup scope.

## Protect deployment and future work

Establish the latest final deployment plan from explicit user decisions and
versioned inference/ensemble/package manifests, not solely the highest score.
Trace the dependency graph to exact runs, folds, seeds and checkpoint variants,
including EMA/raw, best/final, ensembles, teachers, distillation, calibration,
quantization, export, resume sources, rollback and queued jobs. Check active
processes/controllers where accessible; a stale registry status is insufficient.

Protect active or queued consumers, deployed/selected artifacts and their
required source dependencies, rollback commitments, unique useful baselines,
and explicitly pinned runs. If deployment plans conflict or are absent, mark
potentially relevant artifacts **hold** and ask a targeted question while
continuing independent inventory. Do not infer that one strong fold replaces
all folds, or that a compiled export makes its source checkpoint unnecessary.

Compare results only under matching targets, label/split/data versions, cohorts,
metric definitions/directions and training budgets. Include experiment specifics:
architecture, preprocessing, loss, sampler, seed/fold, precision, batch/chunk and
EMA choices. Preserve informative ablations, complementary ensemble members and
valuable failures; distinguish poor performance from invalid evaluation.

## Propose candidates with recovery evidence

Assign each artifact **keep**, **hold**, or **propose-remove** with evidence.
Prioritize large redundant intermediate checkpoints and completed, superseded
runs with no consumers and adequate reproduction records. Retain inexpensive
metadata by default: configs, source evidence, manifests, launch commands,
dependency locks, metrics, logs, predictions when small, and decision history.
Report sizeable metadata separately; do not silently bundle it into checkpoint
removal. Do not propose recursive deletion of a run directory to save its weights.

Assess recovery before recommending removal. Use the reproduction checklist in
the reference. **A preserved config sufficient to rerun the experiment is the
default requirement; exact historical code is not required.** Save the config
actually used, including effective defaults, inherited settings and CLI/wrapper
overrides; a config filename or a small override file depending on unavailable
parents is insufficient. Saved resolved MLflow parameters can supplement the
config. Preserve data/label/split versions and required initialization identities
so the recipe remains meaningful. Record source revisions/snapshots when already
available, but do not hold an otherwise reproducible run solely because its exact
dirty source bytes or historical code are absent.

Distinguish **verified deterministic reproduction**, **approximate reproduction
from preserved config**, and **blocked/unknown**. Approximate reproduction is
acceptable without additional risk approval when the config/recipe is sufficient;
state expected differences and missing historical code without claiming identical
weights. Never mark exactness without an existing repeatability comparison and
all required inputs. Retraining is not restoring the old checkpoint, and deleting
a full checkpoint loses its exact optimizer/RNG/continuation state unless retained
elsewhere. Missing or insufficient configs, unavailable data/pretrained inputs,
and unresolved deployment/active-job dependencies remain holds unless the user
explicitly accepts the specific loss or resolves the dependency.

Produce an immutable versioned Markdown proposal plus machine-readable manifest
using the reference contract. Include a conservative recommended set, optional
higher-risk sets, bytes by storage location, protected artifacts, unresolved
questions and exact actions. List retained reproduction evidence and remaining
recovery costs/limitations per run. When asking for approval, point to the
proposal ID and exact candidate IDs, and explain that this skill requires
explicit user approval before changing any artifacts. End the audit with
**nothing deleted or changed in experiment storage**.

## Only after explicit approval

Approval must identify the proposal version and selected artifact/action IDs (or
explicitly approve the complete named recommended set). For ambiguous replies,
clarify. A changed proposal needs renewed approval for changed actions; approval
is not transferable to replacements with the same path.

Before executing, revalidate the exact approved files/remote versions, file
identity and size/mtime or checksum, containment, dependencies, active consumers,
and preserved reproduction evidence. If identity, dependencies or recovery
assumptions changed, skip the affected item and present the change; do not expand
scope. Record the user's approval and the approved manifest hash. Use only exact
approved paths/object versions; no broad globs, recursive run deletion, symlink
traversal or tracking-service cascade. Stop on unexpected errors and report partial
completion rather than broadening commands or retrying blindly.

Keep an append-only receipt of attempted/completed/skipped actions and actual
reclaimed space. Record deleted checkpoint identity, original metric/weight type,
retained evidence locations, recovery limitations and any errors in project
records without erasing prior history. Verify preserved metadata and protected
artifacts remain. Report what changed and what did not; never call a dry run or
proposed byte total actual space reclaimed.
