# Decisions

This file tracks shared project decisions as SKPv2 is cleaned up into a reusable experiment template.

This is the SKPv2 **template/framework** decision history. Projects cloned from
SKPv2 must keep their modeling, dataset, evaluation, and release decisions in
`docs/DECISION_LOG.md`, with experiment tracking in
`docs/EXPERIMENT_HISTORY.md`. Those project documents are intentionally
separate from this file.

## 2026-11-08: Experiment cleanup proposals

- Add a reusable experiment-cleanup skill that prioritizes checkpoint storage,
  reconciles experiment results with final deployment and queued consumers, and
  preserves inexpensive metadata and reproduction evidence.
- Audits produce versioned proposals and exact artifact manifests. No deletion,
  movement, state stripping or other artifact changes occur without explicit
  user approval of the concrete scope; approved actions require fresh identity
  and dependency checks and an append-only receipt.
- Distinguish approximate reruns from identical checkpoint recovery and exact
  training continuation. Insufficient config/input evidence and ambiguous deployment dependencies
  remain holds, never implicit permission to remove artifacts.
- Default recovery standard is approximate reproduction from the actual preserved
  config plus effective settings, data versions and initialization identities.
  Exact historical code/dirty snapshots are useful but not required; their absence
  alone is not a cleanup blocker. Explicit artifact-removal approval is unchanged.

## 2026-11-08: Deep-review integration corrections

- Count successful optimizer steps for EMA using optimizer post-step hooks;
  gradient-scaler skips do not advance the decay ramp. Persist this count and
  reject scaled historical resumes that lack it instead of guessing.
- Serialize evaluated EMA weights through the module checkpoint hook as well as
  the callback path because Lightning skips callback hooks for weights-only saves.
- Built-in masked BCE/focal and ignored/weighted cross-entropy report their true
  reduction denominators. Combined losses preserve component denominators and
  recompute weighted component means after global aggregation. Unsupported
  non-mean validation reductions fail explicitly.
- Reconstruct legacy fractional validation using Lightning's floored batch
  interval and completed optimizer updates. ETA requires a previously observed
  comparable interval; ambiguous schedules remain unavailable.
- Parse CLI overrides for unset options as safe literals with a string fallback.
  Create best-checkpoint links with filesystem APIs, skip missing selections, and
  preserve existing regular files.
- Complete contract, adversarial, independent and final verification passes under
  the repository-local deep-review skill; retain explicit GPU/NCCL limits.

## 2026-11-08: Pre-commit integration review

- Restrict raw-weight restoration to fit/resume. Standalone checkpoint loading
  and validate/test/predict use inference weights; an existing EMA callback does
  not overwrite an explicitly loaded evaluation checkpoint.
- Broadcast prediction-export serialization/write failures before metric
  collectives so an empty or non-writing rank cannot be stranded.
- Deduplicate tied EMA state aliases, reject changed alias topology, and reject
  non-finite classification metric inputs before threshold mapping.
- Honor explicit OneCycle epochs/steps horizons and verify the initialized
  trainer world size against the schedule/topology contract.
- Add failing-before-fix regressions, standalone Lightning evaluation coverage,
  and a two-rank export-failure probe before preparing the migration commit.

## 2026-11-08: Downstream correctness migration

- Implemented sampler-aware schedule sizing and scheduler-specific dispatch, with
  one shared interval contract; Config deepcopy now preserves recursive graphs
  without sharing mutable nested values with the source.
- New configs enable ordinary EMA (0.99, linear 500-update decay ramp). Checkpoints
  separate evaluated inference state from raw resumable state and persist averaging
  settings. Historical ambiguous EMA resumes fail rather than infer new defaults.
- Exact validation is opt-in and bounded to tested classification contracts.
  Empty-rank control batches keep Lightning hooks aligned without fake predictions.
  Loss reductions expose denominators; undefined mapped-target AUCs stay undefined.
- Added generic opt-in prediction export, curves, provenance scaffolding, seeded
  group sampling, MIL pooling/chunk controls and DICOM inspection/serial loading.
  Kept clinical mappings, project registries and submission policy downstream.
- GCP socket settings now require an explicit profile and preserve user overrides.
- Added the audit-only template-review skill, read-only inventory helper and
  classified backlog. See docs/UPSTREAM_MIGRATION.md for compatibility boundaries,
  accepted contracts and tests. GPU/NCCL and secondary architecture ports are not
  certified by the CPU suite. No wholesale cherry-pick, commit or push was made.

## 2026-11-08

- Added the repository-local training-monitor skill with full and fast modes,
  matched-epoch ablation reporting, schedule-aware budget comparisons, and a
  read-only Linux resource collector. Fast mode avoids process-tree/PSS and GPU
  process-attribution scans and bounds log reads.
- Generalized the downstream monitoring reference to the template's MLflow
  server and artifact conventions. Dataset-specific metrics, paths, and sampler
  assumptions remain in downstream projects; monitoring never changes jobs or
  weakens the template's MLflow launch contract.

## 2026-07-27

- Added a reusable slice-wise 3D multiple-instance learning classifier for
  volume-level supervision without slice labels. The model treats `(B, C, D,
  H, W)` inputs as bags of `D` slices, applies a shared 2D timm encoder, and
  produces both volume-level `logits` and pseudolabel-ready `slice_logits`.
- Chose length-normalized Log-Mean-Exp as the default aggregation so otherwise
  identical volumes do not receive systematically shifted logits solely because
  they contain different numbers of valid slices. Raw Log-Sum-Exp remains
  configurable.
- Added optional `(B, D)` valid-slice masking for padded variable-length bags
  and made all-invalid bags fail explicitly.

## 2026-07-24

- Made MLflow tracking fail-closed for normal training. Missing, local, or
  unreachable tracking backends now abort before the run directory and trainer
  are created.
- Added a per-launch `--allow_local_mlflow` escape hatch. It is deliberately a
  CLI flag rather than a persistent config default so local fallback always
  reflects explicit authorization.
- Added root `AGENTS.md` as the canonical new-project bootstrap and orientation
  contract, including remote setup, comprehensive code/data inspection,
  documentation scopes, MLflow safety, experiment construction, and completion
  checks.
- Added required `docs/EXPERIMENT_HISTORY.md` and `docs/DECISION_LOG.md` starter
  documents to every SKPv2 clone.
- Defined Experiment History as the concise registry of runs, configurations,
  outcomes, and dispositions.
- Defined the project Decision Log as the detailed record of evidence,
  alternatives, tradeoffs, interpretations, and downstream consequences.
- Explicitly allowed intentional overlap between the two project documents.
- Kept root `DECISIONS.md` exclusively for reusable SKPv2 template/framework
  decisions, distinct from each cloned project's `docs/DECISION_LOG.md`.
- Added tests that enforce the presence, identity markers, cross-links, and
  scope distinction of the two project documentation files.

## 2026-06-03

- Initialized Git locally, but intentionally did not make an initial commit yet.
- Renamed the default branch from `master` to `main` for GitHub readiness.
- Initialized `uv` project metadata with project name `skp` and Python `>=3.11`.
- Kept the current source layout unchanged for now. The old package name was `skp`; the directory is currently named `src` to align with modern `uv`/Python packaging expectations. The exact import/package layout still needs a deliberate cleanup decision.
- Adopted the standard `src` package layout: repository source files now live under `src/skp`, preserving `skp` as the importable package name while keeping `src` as the packaging container.
- Added `skp-train = "skp.train:main"` as the future console-script entry point.
- Removed experiment-specific config directories from `src/skp/configs` to keep the template repository uncluttered from the outset.
- Kept only the shared `Config` class plus starter templates for 2D classification, 2D segmentation, 2D classification+segmentation, and 3D segmentation.
- Changed `Config` from silently returning `None` for missing attributes to a strict namespace. Missing attributes now raise `AttributeError`; optional config values should be read intentionally through `cfg.get("name", default)`.
- Added `Config.require(...)` for future explicit validation of required config keys.
- Updated core starter paths to use explicit optional config reads, with remaining legacy task/model modules to be handled as they are cleaned or retained.
- Introduced composable config defaults in `src/skp/configs/defaults.py`. Templates now apply relevant defaults first, then define/override experiment-specific values below.
- Removed prior non-MLflow experiment logging support. MLflow is now the only supported experiment logger.
- Removed experiment-specific dataset directories from `src/skp/datasets`.
- Kept reusable generic dataset loaders and added a custom dataset skeleton under `src/skp/datasets/templates`.
- Added `test` as an accepted dataset mode. Current reusable loaders treat `test` like `val`; `inference` remains the mode for running over the full annotation file.
- Added small dataset helper functions in `src/skp/datasets/_utils.py` for mode validation, annotation loading, mode/fold filtering, transform selection, and collate selection. Dataset-specific sample loading and target construction remain local to each dataset module.
- Standardized reusable datasets on albumentations-style transforms and removed TorchVision transform handling.
- Made dataset annotation loading infer common tabular file types from extension instead of assuming CSV.
- Updated the custom dataset template to mirror the common reusable dataset structure, so it can be copied and edited directly.
- Removed historical custom/stochastic loss modules from the reusable template.
- Kept MONAI loss wrappers and MONAI-backed task features optional by using lazy imports with explicit error messages.
- Added a custom loss template under `src/skp/losses/templates`.
- Audited reusable loss functions for correctness. Dice/Tversky-style losses should return conventional positive losses to minimize, subloss modules must be registered with PyTorch, and class-weighted reductions should normalize by the class weight sum.
- Kept EMA and MLflow system monitoring as core callbacks. Custom GPU stat logging is now opt-in because MLflow system metrics should be the default monitoring path.
- Default checkpoints now save full training state rather than weights only, so optimizer/scheduler/callback state including EMA can be resumed. `cfg.save_weights_only = True` remains available when a lean checkpoint is preferred.
- Kept custom metric calculations rather than relying on built-in TorchMetrics implementations. TorchMetrics remains useful as a state container, while sklearn/PyTorch helper functions provide explicit control over edge cases and competition-specific behavior.
- Reduced core metrics to reusable classification/regression helpers and generic segmentation Dice metrics. Experiment-specific metrics were removed from core and replaced with copy/paste templates under `src/skp/metrics/templates`.
- Reduced `src/skp/models` to reusable core model paths plus templates. The core keeps 2D classification, 2D/3D segmentation wrappers, generic decoders, 3D encoder helpers, and MedNeXt. Old domain-specific and experiment-specific model folders were removed from the reusable template.
- Consolidated 2D timm-backed segmentation on `segmentation.base` and removed the narrower `segmentation.timm_unet` wrapper. The reusable U-Net path should not use the raw input image as a decoder skip connection.
- Removed the stale standalone `segmentation.deeplabv3` wrapper. DeepLabV3+ remains available through generic segmentation wrappers via `DeepLabV3PlusDecoder` and `DeepLabV3PlusDecoder3d`.
- Moved MedNeXt SparK/sparse pretraining code out of `segmentation/mednext` and into `models/pretraining`, keeping `segmentation/mednext` focused on segmentation models.
- Removed legacy standalone `convnextv2_3d`, `convsqueeze3d`, and `efficientnet_3d` model files. Core 3D encoder support now stays concentrated in `models/encoders3d.py` and dedicated 3D segmentation wrappers.
- Removed `models/blocks`; its reusable 3D block helpers were no longer imported by retained models after the legacy 3D encoder cleanup.
- Kept the active sparse MedNeXt pretraining path mask-based rather than adding sparse-convolution library dependencies. Removed the older unused gather/scatter sparse normalization file so sparse logic lives in `models/pretraining/sparse_mednext_blocks.py`.
- Sparse pretraining masks should use an exact per-sample visible patch count, cover border voxels when dimensions are not divisible by patch size, and compute reconstruction loss over the full masked element count including channels.
- Sparse encoder convolutions should zero masked inputs before convolution, preserve the active support for stride-1 convolution blocks, propagate support through resolution-changing convolutions, and re-mask outputs before normalization or downstream layers.
- Channel-wise LayerNorm does not need sparse statistics because it normalizes across channels independently at each spatial location. In sparse blocks, `norm_type="layer"` uses the dense-equivalent channel LayerNorm and only re-applies the mask after affine.
- Simplified `LinearWarmupCosineAnnealingLR` to explicit `init_lr`, `max_lr`, `final_lr`, and either `warmup_steps` or `pct_start`. The optimizer LR remains the default peak LR unless `scheduler_params.max_lr` is set.
- When `scheduler_params.max_lr` is explicitly set, it must match the optimizer's actual param-group learning rates. Warmup percentages are interpreted as a fraction of total optimizer steps after DDP world-size and gradient-accumulation adjustment.
- Removed AdamP from the reusable optimizer core. AdamW remains the recommended default unless an experiment has a specific reason to choose another standard PyTorch optimizer.
- Kept dataloader construction and training samplers under `tasks/` because they are training policy rather than dataset definition. Added a small `BaseTask` for common Lightning plumbing while leaving each task's train/validation logic copy-paste friendly.
- Removed the experiment-specific aneurysm crop classification-segmentation task from the reusable template.
- Made task scheduler handling optional, fixed dataloader worker/WebDataset edge cases, and made 3D gradient norm logging opt-in through `cfg.log_grad_norm`.
- Moved lightweight hyperparameter search utilities from `toolbox/halton.py` to first-class `skp.search`, and moved progressive-resizing/sweep orchestration out of `train.py` into `skp.runners`.
- Cleaned `toolbox/` into named optional utility modules: `cropping`, `cv`, `dicom`, `ensembling`, `images`, `masks`, and `plotting`. Removed duplicated legacy DICOM utilities and the generic grab-bag modules.
- Kept DICOM support optional and lazy. Importing the core package or toolbox no longer requires pydicom/dicomsdl unless DICOM functions are called.
- Added machine-readable config field documentation in `CONFIG_FIELD_DOCS` next to the default values, so ambiguous defaults can be explained in one maintained location.
- Added dependency groups to `pyproject.toml`: `dev` for tests/linting, `train` for the core training stack, `medical` for MONAI/DICOM support, `video` for optional pytorchvideo backbones, and `wds` for WebDataset.
- Added a detailed README covering project structure, config philosophy, training flow, optional features, sweeps, tests, and extension points.
- Added `QUICK_START.md` using ImageWoof-160 as a small real-image classification example that exercises the standard config, `simple2d` dataset, classification task, timm model, optimizer/scheduler, metrics, checkpoints, and MLflow logging path.
- Removed default segmentation decoder types from config defaults. Decoder architecture is experiment-defining, so segmentation configs/templates should set `cfg.decoder_type` explicitly.
- Added `data/` to `.gitignore` for local quick-start datasets and other experiment data.
- CPU-only quick-start commands should set `CUDA_VISIBLE_DEVICES=""` and `--precision 32-true` so the smoke test remains independent of any visible but unusable CUDA installation.
- Environment reporting in `train.py` should not initialize cuDNN unless CUDA training is requested.
- MLflow system monitoring should degrade to a warning when the installed MLflow version does not expose the expected system-metrics monitor API.
- Run IDs should use `petname` in `<word>-<word>-<4digit_int>` format rather than opaque random hashes.
- Dataset configs must explicitly set `cfg.split_column`. K-fold splits use `cfg.fold`; fixed train/val/test splits use `cfg.train_split_values`, `cfg.val_split_values`, and `cfg.test_split_values` with guardrails for missing columns, overlapping values, and empty subsets.
- CLI batch-limit arguments should preserve integer-looking values as integer batch counts. Fractional values such as `0.25` remain percentage limits for Lightning.
- Pin the core training environment to the official PyTorch CUDA 12.4 wheel family (`torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`) with explicit uv source mapping for those packages only.
- GPU/DDP training should fail early when CUDA is unavailable, too few devices are visible, NCCL is missing, or SyncBatchNorm is requested outside CUDA DDP.
- Custom metrics should manually gather prediction/target state across distributed ranks before computing validation scores, preserving explicit metric control while making DDP metrics global rather than per-rank averages.
- Added `.env.example` and automatic `.env` loading so remote MLflow tracking
  URIs and credentials can be configured locally without committing secrets.
- Added a GCP GPU environment cleanup for CUDA runs that forces NCCL onto the
  socket backend, removes inherited `LD_LIBRARY_PATH`, and can be disabled with
  `GCP_NCCL_SOCKET_HACK_DISABLE=1`.
- Removed the default MLflow experiment name from `runtime_defaults`. Experiment
  configs now set `cfg.project` explicitly so cloned projects do not all log to
  the template-level `skp` experiment.
- Added `TEMPLATE_BACKLOG.md` as a slow lane for small reusable template improvements that should be batched instead of committed and pushed one by one.
- MLflow run names now use `<config_name>/<run_id>` while local artifacts keep
  the existing `experiments/<config_name>/<run_id>/...` layout, making shared
  experiments easier to scan without opening run details.
- CUDA training now defaults to `--devices -1`, resolving to all visible CUDA
  devices during trainer validation before Lightning trainer construction.

## 2026-11-08: Reusable deep-review skill

- Include `.agents/skills/deep-review` in the template so downstream clones inherit
  contract tracing, failure-path verification, fresh-context review and explicit
  readiness criteria within one review request. Keep template-review focused on
  upstream migration. Neither workflow claims exhaustive bug detection.
- The skill passed structural validation and an isolated independent review trial
  that found an integration regression despite passing tests, preserved source,
  and distinguished an undocumented contract from a confirmed defect.
