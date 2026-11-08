---
name: experiment-cleanup
description: Audit experiment storage, results, reproducibility and deployment dependencies to propose checkpoint cleanup. Use when experiments accumulate or storage needs reclaiming. Never remove or alter experiment artifacts without explicit approval of a concrete cleanup proposal.
---

# Experiment cleanup

Build a small, justified checkpoint keep set, then propose removing the rest.
Assume most exploratory experiments will never be rerun. The burden of evidence
is on retaining large weights, not proving each experiment obsolete. Preserve
inexpensive configs, results and history even when its checkpoints go.
Approximate reproduction from a sufficient config is enough; exact historical
code and identical weights are not required. This is a proposal policy, never
authorization to delete anything.

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

## Establish the minimal keep set

Start from the user's current final selection and actual future needs. Identify
exact model members, folds, seeds and weight variants required for deployment,
plus helper models and genuinely required dependencies. Verify active jobs or
concretely committed queued work against controllers/processes where accessible.
A hypothetical future use is not an active dependency.

Require a present-tense reason for every retained checkpoint:

- needed by the current final model or deployment pipeline;
- needed by a verified active job or concretely committed next action;
- an explicitly requested rollback version or currently reaffirmed pin;
- one of a small, named set of reference/posterity models with distinct value.

Propose a few reference models at most by default, with a reason for each;
do not retain every architecture, ablation, failure or former best. Preserve
those experiments' scientific information through configs, metrics, predictions
when small, logs and decision history instead. Ask about a larger reference
collection only if a concrete project need supports it.

Historical `selected`, `exported`, `pending` or `running` labels, an old
"checkpoints retained" note, recent experiment numbers, and old package/script
references are not automatic retention reasons. Reconcile them with the latest
user decision. Distinguish a historical record of preservation from a current
instruction to keep an artifact. If an explicit standing pin genuinely conflicts
with the current selection, isolate that conflict for clarification; do not carry
forward every old pin as a permanent exemption.

For each kept model, choose the minimal useful artifacts. Required ensemble
folds remain distinct, but keeping a final inference model does not automatically
justify best/raw/final/EMA duplicates, every export copy, or optimizer state.
Retain full training state only for an identified continuation need or explicit
preference. Where an export suffices, assess whether its source weights are
actually needed for rebuilding or changing the supported deployment. Propose
state stripping or replacement separately when needed; never perform it in an
audit. Do not count training lineage or old evaluation inputs as live dependencies.

If the final selection is unclear, infer a provisional shortlist from current
evidence and ask one focused clarification. Hold only artifacts directly affected
by that uncertainty. Continue proposing cleanup for the rest; do not protect all
historical families or all recent runs while waiting.

Compare results under matching targets, data/split/label versions, cohorts,
metrics and budgets when choosing reference models. Distinguish poor performance
from invalid evaluation, without assuming either needs its weights preserved.

## Propose cleanup outside the keep set

Assign each artifact **keep**, **hold**, or **propose-remove**. Every keep needs
a concrete retention reason; every hold needs a specific unresolved question and
a condition that would resolve it. Once scope and role are established, propose
removing checkpoints outside that justified keep/hold set. An explicit historical
rejection or proof of inferiority is not required. Do not relegate most unselected
runs to an optional list solely because they were never formally marked rejected.

Retain inexpensive metadata by default: actual configs/effective settings,
source evidence when available, manifests, commands, dependency locks, metrics,
logs, small predictions and decision history. Report sizeable metadata separately.
Do not propose recursive run-directory deletion to save its weights.

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
elsewhere. Missing or insufficient configs require a bounded attempt to preserve
available effective settings. If that fails, disclose the loss of reproducibility
and request explicit acceptance for those items. Missing historical data or
initialization affects recovery fidelity; it is not a reason to preserve an
otherwise unused checkpoint indefinitely. It may remain a removal candidate with
that limitation prominently disclosed for approval. Only unresolved live
consumption, identity/scope, current pin or necessary recipe questions warrant
specific holds; do not turn a reproduction audit into a blanket retention rule.

Produce an immutable versioned Markdown proposal plus machine-readable manifest
using the reference contract. Lead with the proposed keep set and its reasons,
then the recommended removal set covering the remaining eligible checkpoints.
Show total storage before and after, split into kept weights, held weights and
preserved metadata. Explain any large remainder: a proposal that retains most
checkpoint storage requires concrete current reasons, not generic caution.
Include material recovery limitations, unresolved questions and exact actions. List retained reproduction evidence and remaining
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
