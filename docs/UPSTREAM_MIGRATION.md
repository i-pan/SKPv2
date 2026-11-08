# Downstream migration contracts

Implementation baseline: template `bd42b9c60fc36e53cb5767be66e8646bb7480e0d`;
downstream `c6dd63f7264edaafb8603d4a565e2ba35b837280`, including existing dirty
source. Work is generic and lives in the template. No downstream experiment code
or historical data was copied wholesale. Classified findings are in
[UPSTREAM_FINDINGS.json](UPSTREAM_FINDINGS.json), with disposition in
[TEMPLATE_BACKLOG.md](../TEMPLATE_BACKLOG.md).

## Schedule and reproducibility

Scheduler construction uses the selected sampler length, DDP padding, loader
`train_drop_last` (default true), batch limits and ceiling division for partial
accumulation. Set `scheduler_interval` explicitly; the shared fallback is epoch,
matching historical task execution. Only cosine, warmup-cosine and OneCycle
receive an inferred horizon. Explicit `T_max`/`total_steps` remains authoritative, as does OneCycle's explicit
`epochs` × `steps_per_epoch` horizon.
StepLR and ReduceLROnPlateau receive only their own arguments. Plateau requires
epoch validation. Resolved batches/updates/horizon are printed before training.
Iteration counts require an iteration sampler or WebDataset; they cannot silently
resize a standard loader. Dynamic accumulation dictionaries are unsupported.

`seed` opts into Lightning worker seeding. `fold_seed_stride` (default 0) makes
fold-dependent seeds explicit. `required_world_size` rejects topology drift; the initialized trainer topology is
also checked against the configured schedule before training starts.
Optimizer batch is batch × world size × accumulation. Encoder normalization
batch/chunk is logged separately; for volume models, derive the actual flattened
slice count from input depth. SyncBatchNorm further changes the normalization
population. An optimizer effective batch is never a BatchNorm batch guarantee.

`SKP_ENV_PROFILE=gcp_socket` supplies unset socket defaults only. Existing NCCL
settings and `LD_LIBRARY_PATH` are preserved. No profile is active by default.

## EMA and checkpoints

New `runtime_defaults` enable ordinary EMA at decay 0.99, a 500-update linear
ramp, no delay, every optimizer update. Update 1 copies current weights (decay
0), update 500 reaches 0.99. Accumulation microbatches and validation do not
advance EMA. Successful optimizer post-step hooks also exclude gradient-scaler
overflow skips; the successful-update counter is saved independently of Lightning's
step count. Scaled historical checkpoints lacking this counter require migration. Set `cfg.ema = {"on": False}` to disable it. Switch EMA remains an
explicit `switch_ema=True` choice. Old explicit `use_warmup` configurations use
the historical power formula and zero-based decay step convention.

`state_dict` in best and last checkpoints contains inference weights. With EMA
these are the EMA weights evaluated by validation. `prediction_weights` says
`ema` or `raw`; `averaging` records parameters and update count. A full checkpoint
also contains `raw_state_dict`, optimizer/scheduler state and callback state.
`BaseTask.on_load_checkpoint` selects raw weights only for Lightning fit/resume.
Standalone `load_from_checkpoint` and Lightning validate/test/predict keep the
inference `state_dict`. EMA callbacks do not replace explicitly loaded inference
weights during standalone evaluation. `save_weights_only=True` is not a
resumable checkpoint, but still saves the evaluated EMA inference state through
the module hook. `ema_weights.pt` is a convenience final inference artifact;
it is not necessarily the best checkpoint.

Saved EMA settings override new defaults on resume. Tied state aliases are updated once,
including when buffer averaging is disabled. Historical EMA checkpoints
without settings fail closed: migrate a copy using the original config, verified
optimizer/update count, and original raw weights before resuming. Do not guess
how many old microbatch updates occurred or silently restart the decay ramp.
Use Lightning `Trainer.fit(task, ckpt_path=...)` for a full-state resume; loading
pretrained weights is a new training run. Existing project wrappers remain
project-owned and must be retired deliberately rather than layered onto this
new callback. Multiple automatic optimizers are unsupported by EMA.

## Exact evaluation and prediction export

`exact_validation=True` is opt-in for map-style classification with the standard
AUROC/AVP/MAE/MSE/Accuracy/Kappa/QWK metrics. Other task/metric combinations are
rejected until their empty-rank behavior is tested. DDP buffer broadcasts happen
once at validation start, not on uneven forward calls. Model-internal forward
collectives must not run in this mode. SyncBatchNorm with running stats uses
those stats during evaluation. GPU/NCCL integration has not been verified.

The sampler never pads validation indices. An empty rank gets a control batch
which runs no model, metric, loss or prediction update but keeps Lightning's
epoch-end collective order aligned. Datasets must expose unique `sample_ids` and
batches must contain corresponding `sample_id` strings. Exact coverage is checked
at the end of each full validation pass, outside sanity checks. Partial validation
limits are rejected. The simple2d dataset supports `sample_id_column` explicitly;
use an immutable source identifier, not an index that changes after filtering.

`prediction_export=True` installs a metric-independent callback. Classification
returns logits, targets and IDs; custom tasks can implement the same contract.
Set `target_order` and `data_provenance` (manifest ID/path/hash, split and label
versions). Export writes atomic NPZ files under each split's `predictions/`,
ordered by dataset sample IDs, with fold, epoch, step, identity activation and
weight type. Duplicate, missing, and unexpected IDs fail. Rank-zero serialization/write errors
are broadcast so every rank exits before entering subsequent metric collectives. Current export uses
CPU object gather and holds one validation pass in memory; it is intended for
modest prediction tables, not large segmentation maps. Export is opt-in and is
independent of the knee task or any competition cohort definition.

Generic scalar losses default to sample-weighted reduction. For masks, weights
or mixed heads, return `out['loss_denominators'] = {'loss': denominator, ...}`;
each scalar is multiplied by its denominator, summed across batches/ranks, and
divided by total denominator. Built-in ignored/weighted CE and masked sequence
BCE/focal provide their actual denominators, including zero for all-masked batches.
CombinedLoss propagates component denominators and `loss_compositions`; combined
validation loss is the weighted sum of globally reduced component means, not a
batch-size-weighted average of mixed-denominator batch totals. Each combined head
requires a unique output key; repeated keys previously overwrote a requested loss
and are now rejected. These metadata changes preserve training scalar semantics,
including existing NaN empty means; zero denominators exclude them from validation.
Non-mean built-in
validation reductions fail explicitly rather than producing a plausible wrong mean. Nonlinear cohort metrics such as Dice are not
reconstructed from this loss aggregation. `masked_soft_bce` returns loss,
numerator and denominator, with `weight` or `element` normalization and a safe
zero loss for all-masked batches. A fully masked validation cohort has undefined
loss (NaN), not a measured zero.

AUROC/AVP map targets at `metric_target_threshold` (default 0.5) before diversity
checks. Non-finite targets/predictions are rejected rather than mapped to labels.
Undefined classes return NaN, with explicit valid/undefined class counts.
The macro mean remains NaN if any required class is undefined; it never silently
changes the set of labels being measured. Group probability aggregation supports
explicit mean/max; pooled OOF metrics and mean fold scores must be reported as
different quantities.

## Optional workflow tools

`skp-curves RUN ... --experiment-name NAME --tracking-uri URI -m METRIC ...`
supports multiple runs/metrics, fractional epochs, step axes, watch mode, observed
validation ETA and best/final summaries. Legacy fractional schedules use floored
batch intervals and completed optimizer updates; ambiguous duplicate-step checks
remain unavailable. ETA waits for an observed interval comparable to the next
interval, including longer epoch-wrap gaps. It has no project-name, compartment
mapping or workstation URI default. `MLFLOW_TRACKING_URI` can supply the URI.

`skp.provenance` and `scripts/provenance.py` provide opt-in scaffolding:

- immutable dataset manifests with component hashes and label/split versions;
- source archives containing actual tracked, dirty and nonignored untracked bytes;
- append-only planned/running/completed/failed/interrupted experiment events,
  with exact commands, config/source/dataset references and closing dispositions;
- optional immutable prompt-ID registration and result bindings to experiment,
  prompt hash, model revision, rendering inputs, decoding parameters and artifact.

Use a source-only checkout or appropriate ignore rules for snapshots. Secret-like
paths and symlinks fail rather than being silently copied. Inspect inventories
for private data/secrets first; filename checks are not a secret scanner. Snapshot
archives are evidence, not an automatic restore or upload mechanism. Consumers
must save prompt-result bindings, verify hashes before/after generation and add
MLflow IDs, checkpoints, validation and external scores to registry events. These
utilities intentionally do not launch work or impose LLM workflows on projects.
The downstream competition's richer registries remain downstream.

## Optional components

`GroupBalancedSampler` consumes `dataset.group_ids`, `samples_per_group` and
`seed`; it selects per-group examples deterministically by epoch. simple2d exposes
IDs through `group_id_column`. Distributed wrappers propagate `set_epoch`.

MIL defaults remain normalized LSE. `mil_pooling` can select `attention`,
`label_attention`, or `gru`. GRU masks must be nonempty valid prefixes. Slice
logits remain available for auxiliary supervision. `encoder_chunk_size` is an
explicit training/normalization parameter, not solely a memory knob. Chunked DDP
training rejects differing flattened slice counts or chunk sizes before encoder
collectives. Inference parity is numerical within tolerance, not bitwise across
precision or kernel choices. The CPU tests cover components; real backbone,
pretrained weights, GPU precision and NCCL still need project-specific smoke tests.

DICOM utilities add serial header inspection, deterministic file/geometry ties,
explicit serial loading (`max_workers=0` or 1), and nearest-plane classification.
No knee filtering, laterality or crop policy is imported. The full loader retains
its existing filtering policy; header inspection does not filter anatomy.

## Verification boundaries

Regression tests exercise CPU Lightning fit/checkpoints, accumulation 1/2,
partial batches, EMA ramp/resume state, two-process gloo exact evaluation with
uneven/empty shards, coverage rejection, metric reconstruction, masked reductions,
chunk parity, provenance tampering and dirty inventory. No real data preprocessing,
training experiment, MLflow run, GPU benchmark, cloud job or submission is launched.
Optional CSN/R(2+1)D and Swin pretrained/checkpointing migrations remain gated on
architecture/shape and pretrained-load tests. Template test success does not
establish competition performance or untested GPU behavior.

## Verification result (2026-11-08)

- Full template suite: **131 passed** in 28.03 seconds.
- Repository-wide `ruff check .`: passed.
- `git diff --check`: passed.
- Skill frontmatter/scaffold validator: passed.
- `python -m skp.curves --help` and provenance CLI help: passed.
- Bounded read-only inventory compared both actual worktrees; files above 8 MiB
  are explicitly marked unverified, not silently treated as identical.

The CPU tests include a full checkpoint resume compared numerically with an
uninterrupted run, EMA/BatchNorm through uneven and empty rank validation, and
collective rejection of unequal encoder chunk shapes. Test-created training
processes finished. No production training, MLflow run, commit, push or deployment
was performed. Existing downstream processes were not changed.

## Second review and commit preparation

The second review corrected inference-versus-resume checkpoint selection, tied
EMA updates, non-finite metric target mapping, rank-zero prediction export failure
propagation, explicit OneCycle horizons, and initialized topology validation.
The checkpoint, tied-weight and non-finite-target probes failed before their
fixes; new regressions cover every correction. A two-rank probe confirms export
errors reach an empty rank, and Lightning standalone validation reloads the saved
inference state even with an existing EMA callback.

Final verification: **145 tests passed** in 45.96 seconds; repository-wide lint,
whitespace checks, skill validation and CLI help checks passed. GPU/NCCL and
secondary pretrained-architecture gates remain open. The template changes are
prepared for a local commit; no commit, push, deployment or production training
launch was performed during this review. Test processes finished.

## Deep-review completion

The requested deep-review covered contract tracing, adversarial integration
probes, fresh-context reviews of training, evaluation and workflow code, fixes,
and independent cross-review of the final affected contracts. It found and fixed
weights-only EMA serialization, FP16 skipped-update counting, built-in loss
reductions and CombinedLoss metadata, fractional curve timing, unset CLI option
types, and best-checkpoint link handling. Regression tests distinguish raw from
EMA weights, skipped from successful steps, and unequal component denominators.
A temporary-file smoke also exercised provenance CLI prompt registration.

The reviewed scope is the complete template migration from `bd42b9c`, including
the repository-local deep-review skill. No finite review proves absence of bugs.
GPU/NCCL, real backbone/pretrained integration and production dataset behavior
remain outside this CPU verification. Legacy curve reconstruction deliberately
rejects ambiguous schedules; custom schedules need explicit epoch logging.

Final deep-review gate: **168 tests passed** in 63.09 seconds. Repository-wide
ruff, staged/unstaged whitespace checks, both skill validators and CLI help
checks passed. The suite includes a real three-rank gloo test of composed loss
reductions with an empty shard, and CPU GradScaler skip/resume equivalence.
No remaining material findings within the reviewed scope. All review/test
processes finished; no production training, MLflow run, deployment, commit or
push occurred. Changes and the commit message are prepared locally.
