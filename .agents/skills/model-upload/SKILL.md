---
name: model-upload
description: Propose versioned ML model releases for Hugging Face or private local storage, including model cards, inference and fine-tuning access, provenance, and verification. Stop for explicit user review and approval before executing the proposal. Use for model packaging, release preparation, and model uploads, not training or dataset publication.
---

# Model upload

Produce an auditable, usable model release, not just a weight dump. Hugging Face
is the primary remote destination. Private models default to a dedicated local
release folder with the same documentation and verification; private HF hosting
is a separate, explicit choice.

## Establish the release contract

### Mandatory proposal and approval gate

Every new release request starts in **proposal-only mode**, even when phrased as
"upload the model." Inspect existing evidence using bounded read-only checks,
then present a concrete proposal and stop the turn for user review. Invocation,
past releases, confirmed repository names, and visibility preferences are not
approval of a new release proposal.

The proposal must identify selected artifacts/states, inclusion and exclusion
lists, sizes (measured or estimated), destination and visibility/access controls,
source revision, documentation/model-card contents, licensing status, inference
and fine-tuning interfaces, proposed code changes, verification steps, unresolved
risks, and the exact actions approval would authorize. Draft documentation may
be shown in the response; save a proposal file only if the user requests it.

Before explicit approval of that proposal, do not edit project code or release
documentation, export/copy/build artifacts, install dependencies, run model jobs,
create staging/release folders or remote repositories, commit/push, upload, tag,
change permissions/visibility, or finalize private-local storage. Private-local
delivery has the same approval gate as HF publication.

After the user explicitly approves the presented proposal, execute only its
approved scope. Approval of preparation alone does not authorize upload or final
delivery. If the destination, visibility, licensing, artifact selection, or
other material scope changes, present the revised proposal and wait again.
Do not repeat approval requests for unchanged, already-approved actions.

- Distinguish **prepare only**, **local private delivery**, and **HF publication**.
  Preparing a release does not authorize publishing, changing visibility, pushing
  source code, or uploading data. Carry confirmed preferences into the proposal.
  If destination or visibility is missing, list it as an unresolved decision;
  do not begin local packaging while waiting.
- Inspect project instructions, repository role/status, existing export code,
  selected experiments, and actual artifact consumers. Preserve unrelated edits.
- Identify exact components, folds, architecture variants, raw/EMA/SWA state,
  checkpoint selection criterion, and preprocessing/postprocessing dependencies.
  Do not silently substitute a better-looking state for historical training
  inputs. Distinguish training-crop state from deployment state when they differ.
- Establish source revision, storage budget, target platform, release ID, and
  compatibility boundaries. For ensembles, define deterministic subset presets
  only when useful/requested; explain whether they select folds or architectures.
  Never attach full-ensemble metrics to unmeasured subsets.

Read [release-contract.md](references/release-contract.md) for both delivery modes.
Read [hugging-face.md](references/hugging-face.md) only for HF publication.
Consult [sources.md](references/sources.md) when updating this workflow or needing
current provider requirements; verify mutable API/limit details before use.

## Build and document (only after proposal approval)

1. Resolve checkpoint selection from evidence and inspect redistribution terms
   for weights, code, data, and pretrained dependencies separately. Do not infer
   permission to license weights from the source repository license. Escalate
   unresolved rights before publication; do not present a legal determination.
2. Export into a new versioned staging directory. Prefer SafeTensors for PyTorch
   tensors and appropriate non-executable formats for other frameworks. Keep
   optimizer/trainer states out of inference releases unless explicitly required.
   Safe tensor formats do not make accompanying Python code trustworthy by default.
3. Use an explicit inventory/allowlist. Exclude credentials, private paths, patient
   identifiers, training samples, feature caches, logs, and build debris unless
   specifically required and authorized. Inspect actual packaged wheel/archive
   contents, not merely their filenames. Do not recursively upload a workspace.
4. Write the model card and release manifest from observed evidence. Include
   compact source/reproduction links near the top and an inference-first layout:
   installation, usage, presets, outputs, performance and limitations precede
   detailed training, fine-tuning and provenance. Keep the model-specific training
   procedure easy to find through a section link. A buried
   provenance link or dataset-only summary is not sufficient training documentation.
   Include single-checkpoint loading and fine-tuning instructions, not just ensemble
   inference. Identify intentional freezing, input/target contract, and whether
   optimizer resume is supported. Prefer native library save/load conventions;
   custom models need a versioned, installable loader and dependency specification.
5. Record per-file byte sizes and streaming SHA-256 hashes for weights and runtime
   code/config artifacts. Record source commit and any dirty-source snapshot/hash;
   never label a wheel built from changed code as an older clean revision.

## Verify and deliver (only within approved scope)

- Run relevant tests, strict checkpoint loading, and a bounded real inference
  smoke through public entry points. Verify example commands, shape/label order,
  geometry mappings, and supported ensemble selections. Hash validation alone is
  not runtime validation. Do not launch expensive training or GPU sweeps as an
  implicit packaging step; report unavailable checks as release limitations.
- Test a clean install when practical and authorized, isolated from the working
  environment. Verify local-path and remote-ID loading as applicable, plus a
  selected checkpoint for fine-tuning. Private-local smoke should work without
  hidden downloads; document required external dependencies if not self-contained.
- Before publication, show exact payload totals and component/state inventory,
  intended destination/visibility, exclusions, measured results, and open gates.
  Block on secret exposure, broken loading, unresolved rights, or wrong artifacts.
- For local delivery, stage on the destination filesystem and atomically rename
  into a new nonexisting version folder after validation. Do not overwrite prior
  releases or share mutable hardlinks to research artifacts. Record local access
  controls and location; a folder alone is not proof of confidentiality. No Hub
  calls or automatic synchronization in local-only mode.
- For HF, follow the HF reference. After either delivery, verify the delivered
  inventory/hashes, document exact immutable destination, and preserve rollback
  information. Report what was stored, committed, pushed, uploaded, or not done.
  Do not say released when only staging or transport has completed.
