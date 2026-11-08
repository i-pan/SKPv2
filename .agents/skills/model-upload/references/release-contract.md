# Shared release contract

This is a project release convention informed by the sources reference, not a
claim that every provider mandates this exact directory structure.

## Model card

Use `README.md` for both HF and private-local releases. Include applicable fields;
mark missing evidence as unknown/not measured, rather than inventing values.

Place source-code and reproduction-guide links directly below the title or short
overview, before installation/inference examples. Link experiment history and
decision records there when available. For private releases, use authorized
internal references or bundled documentation; explicitly state when source is
unavailable rather than inventing a public repository.

Include a dedicated, model-specific **Training procedure** section before
inference examples. For multi-stage pipelines, explain each released component's
training and dependencies, not only the overall dataset. Cover inputs and
preprocessing, targets, split/leakage controls, initialization and freezing,
sampling, loss, optimizer/LR schedule, batch size and accumulation, epoch budget
and stopping/selection rule, augmentations, and raw/EMA state as applicable.
Use effective run records when available; distinguish reproduction-config values
from verified historical settings and distinguish epoch budgets from selected
epochs. Mark unavailable facts explicitly. Link exact configs and reproducible
commands; links supplement rather than replace the on-card training explanation.
Do not mix discarded ablation recipes into the final model's training account.

Check the actual rendered or retrieved card before declaring documentation
complete: can a new reader immediately find the training code and understand
how these specific weights were trained? A heading or keyword check alone does
not establish this. For card-only revisions, keep file inventories/checksums and
delivery receipts consistent without unnecessarily rebuilding unchanged weights.

- Identity: model/release name, authors/owner, task, version, architecture,
  component inventory, parameter counts if measured, source revision, contact.
- Intended users and uses; out-of-scope uses, foreseeable misuse, domain-specific
  safety limitations, human oversight, and deployment assumptions.
- Training: dataset provenance/access, populations/domains, annotation coverage,
  splits/grouping, leakage controls, pretrained lineage, training recipe and
  reproduction links. Private data can be described without exposing identifiers.
- Evaluation: exact selected state, population/sample counts, split protocol,
  metric definitions and aggregation, thresholds/calibration, uncertainty and
  subgroup results where available. Distinguish validation, test, external, and
  clinical evidence. Report important gaps and known failure modes.
- Inference: accepted inputs, normalization/windowing/tokenization, dimensions,
  coordinate conventions, missing-input handling, output ordering/units,
  preprocessing and postprocessing, companion packages and version compatibility.
- Installation: tested Python/platform/accelerator/dependencies; runnable local
  and (where applicable) pinned-HF examples. Measured latency/VRAM and conditions,
  or explicitly unmeasured. Do not imply hosting an artifact enables hosted inference.
- Fine-tuning: exact weight paths and single-component loader, architecture/config,
  train/eval mode, frozen parameters, loss/target requirements, output-head changes,
  required upstream features, and whether optimizer state is intentionally absent.
- Licensing/access: distinguish code, weights, upstream assets, and data terms.
  Include attribution/citation and relevant notices; do not fabricate citations.
- Limitations: bias, domain shift, robustness, privacy/security considerations,
  untested presets, and maintenance/reporting route. A disclaimer is not evidence
  that a high-stakes system is safe.

## Machine-readable record

Reuse an existing project schema if adequate. Otherwise record:

- Schema/release versions and candidate/released status.
- Immutable source identity and build/dependency identity.
- Component identifiers, relative artifact paths, folds, state type, checkpoint
  selection, label order, ensemble policy, required companion release versions.
- Relative file inventory with sizes and SHA-256 for all payload files except
  the inventory itself (avoid self-referential hashes).
- Evaluation evidence references and validation results/known gaps.
- Delivery destination; remote revision goes in a separate receipt generated
  after upload, avoiding a circular self-referential commit identifier.

Use relative paths; do not leak workstation paths or private experiment records.
Hash code/wheels/configs as well as tensors. A manifest records selection;
verification must actually reject mismatches before consuming artifacts.

## Portable folder shape

Example only; retain framework-native naming when supported:

```text
<model>/<release>/
  README.md
  release.json
  provenance.json
  LICENSE / applicable notices
  requirements or lockfile
  weights/ or native model shards
  configs/ (if not embedded safely)
  runtime package and examples
  verification report
```

Private-local releases need the same card and reproducibility detail. Keep
restricted evidence outside broadly accessible folders; label access limits.
If a local release depends on a private source checkout, preserve an authorized
runtime snapshot or explicitly document that dependency rather than claim a
self-contained archive.
