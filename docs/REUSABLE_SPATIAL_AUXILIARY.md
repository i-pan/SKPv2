# Reusable spatial classification and auxiliary segmentation

These generic components were extracted from downstream ICH work. They do not
encode clinical labels, HU ranges, cohort membership, or architecture-specific
stride changes. Existing models and samplers retain their defaults.

## Spatial head and shared decoder

Select `model="classification.spatial_auxiliary"`. The timm encoder must support
`features_only=True` and return an NCHW feature pyramid. Set `backbone`,
`pretrained`, `num_input_channels`, `num_classes`, and the usual normalization.
Optional `encoder_kwargs` are passed to timm (for example `out_indices` or
`output_stride` where that backbone supports them). This is a new model and state
dict layout, not a drop-in loader for historical downstream checkpoints.

- `spatial_pooling`: `max` (default) or `lse`.
- `spatial_temperature`: positive finite float, default 1.0. LSE is
  `T * log(mean(exp(logit_map / T)))`, computed stably in float32.
- `auxiliary_output_channels`: 0 (default, disabled), or the number of mask channels.
- `auxiliary_decoder_channels`: one width per encoder skip, ordered deep to shallow.

The classifier applies a 1x1 linear projection **before** pooling. Forward returns
`logits` and `logit_map`. Set `return_segmentation=True` for full-input-resolution
`segmentation_logits`. With `return_loss=True`, an enabled decoder is evaluated
and the configured criterion sees both outputs. Ordinary inference skips the
auxiliary decoder. `return_features=True` adds the pyramid. The standalone
`skp.models.spatial.SpatialLogitHead` and `AuxiliaryDecoder` can also be attached
to a downstream encoder; the decoder supports equal-stride skips and odd sizes.

Example model/loss settings (dataset and metrics remain project-specific):

```python
cfg.model = "classification.spatial_auxiliary"
cfg.backbone = "resnet18"
cfg.pretrained = False
cfg.num_input_channels = 3
cfg.num_classes = 2
cfg.spatial_pooling = "max"
cfg.auxiliary_output_channels = 1
cfg.auxiliary_decoder_channels = [128, 64, 32, 16]
cfg.loss = "auxiliary.SpatialAuxiliaryLoss"
cfg.loss_params = {"bce_weight": 0.5, "dice_weight": 0.5, "dice_reduction": "sample"}
```

The batch contains `x`, classification `y`, `segmentation_target` (N,C,H,W soft
probabilities), and `positive_samples` (boolean N-vector). Optional
`segmentation_weights` and `segmentation_mask` broadcast to the target shape.
Weights normalize BCE by their sum, not image area; multiplying all weights by
a constant does not change the loss. The boolean mask removes invalid targets.
Dice uses explicit sample flags, never an inferred threshold on teacher targets.
`sample` averages Dice across selected samples; `batch` computes one ratio over
all selected pixels. Empty selections produce differentiable zero. Dice is
unweighted; BCE ROI weights do not implicitly restrict Dice. These are soft
training losses, not thresholded positive-slice or volume evaluation metrics.
Validation logs use explicit denominators; batch Dice reports a mean of nonempty
batch losses and depends on batching, not a global volume Dice. Under DDP, batch
Dice is local to each rank. Spatial-logit-map supervision and target resizing
remain downstream choices; the provided combined loss supervises classification
and decoder outputs only. Do not enable classification-task mixup with auxiliary
masks unless the downstream task also mixes the masks and sample-selection rule.

## Sampling and validation resources

`sampler="CyclingGroupBinSampler"` requires aligned dataset `group_ids` (nonempty
strings) and `sampling_labels` (binary vector). One sample per group is selected
per epoch. `positive_sampling_probability` defaults to 0.5; empty bins fall back
to the other bin. Each group's positive and negative bins have independent
shuffled cycles, exhausted before reuse. Epoch replay uses the resolved seed and
requires unchanged row order, labels and groups. Repeated iteration of the same
epoch is idempotent. Resume reconstructs bin state by replaying prior epochs;
only the latest plan and per-bin remaining rows are kept in memory.

The normal sampler-aware scheduler uses the number of groups, batch size, DDP
padding, accumulation and configured epoch count. This changes the optimizer-step
budget when changing samplers: choose the intended epoch count explicitly. Use
`train_drop_last=False` for complete epoch coverage. DDP's standard wrapper pads
when group counts are indivisible by world size; exact once-per-group coverage
requires divisible counts. Existing limit-train-batches settings can truncate
coverage. No downstream schedule constants are imported.

`val_num_workers`, `val_persistent_workers`, and `val_prefetch_factor` independently
override training settings. `None` inherits; zero workers is honored for both
ordinary and WebDataset loaders and omits persistence/prefetch arguments.
`prefetch_factor=None` uses 2 ordinarily or 4 for WebDataset.

## Segmentation initialization and frozen encoders

`segmentation.base.Net` now uses the shared normalization helper (including
linear scaling and legacy aliases). Unknown modes raise instead of silently
passing inputs unchanged. `freeze_encoder=True` freezes parameters as before;
`frozen_encoder_eval=True` additionally keeps encoder normalization buffers and
stochastic layers in eval mode whenever the parent enters train mode.

`pretrained_load_order="model_last"` preserves historical precedence. Opt-in
`"components_last"` loads the full model first, then explicitly specified encoder
and decoder checkpoints. This applies to `segmentation.base.Net` initialization;
loading a complete state dict afterward intentionally supersedes initialization.
The legacy `unet_cls` wrapper's `load_pretrained_segmenter` is such a subsequent
full-state load. Prefer one full-model source rather than conflicting wrapper
and inner-model initialization settings.

## Isolation and export integrity

Call `skp.provenance.assert_group_isolation(protected_groups, source_groups)`
before training or pseudo-label creation. `source_groups` maps each teacher or
ancestor name to its training group IDs. All IDs must be stable nonempty strings;
provide patient/entity-level membership where multiple items share a group.
The helper checks every supplied source, rejects overlap, and avoids printing
identifiers. It cannot discover missing ancestors; callers must supply the full
lineage. It is an explicit preflight utility, not an automatic dataset filter.

Best/final prediction export embeds checkpoint SHA256 and checks that the source
checkpoint has not changed by publication time. Every NPZ export has an atomic
`.npz.json` sidecar containing the archive hash and checkpoint hash (null for
in-training predictions without a saved checkpoint). NPZ and sidecar replacement
are not a single filesystem transaction: readers must call
`verify_prediction_export(path, checkpoint_path=None)` before reuse. Missing or
stale sidecars fail closed. Supply `checkpoint_path` to detect a checkpoint
replaced under the same filename. This is integrity checking, not a signed
provenance guarantee; use a single writer per export path. Existing NPZ consumers
remain compatible, but old exports without sidecars are not verified by this API.
