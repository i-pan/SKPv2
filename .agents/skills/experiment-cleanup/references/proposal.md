# Proposal and reproduction contract

## Evidence needed per experiment

Use links or paths with revisions/hashes where available. Mark missing evidence;
do not synthesize an exact config from the current defaults.

- Stable experiment/run ID, tracking ID, status and exact producing command,
  including wrapper, environment/CLI overrides and effective resolved config.
- Best available source revision/snapshot, lockfile/environment, library/runtime
  versions, hardware/topology and precision. Historical dirty source bytes are
  useful evidence, not a default prerequisite for removal. A sufficient preserved
  config/recipe supports acceptable approximate reproduction without exact code.
- Immutable dataset, preprocessing, label and split versions/hashes; availability
  of their components, not merely a manifest. Training input/pretrained checkpoint
  versions and their retained locations. Optional generated-label prompt/model/
  rendering/decoding provenance when the project uses generated labels.
- Seeds and fold routing, sampler/augmentation settings, accumulation, effective
  batch, encoder chunk/normalization settings, training duration/stopping rule,
  scheduler, optimizer and EMA settings. Identify nondeterministic dependencies.
- Results with metric definitions, cohort/target order, checkpoint epoch/step and
  raw/EMA identity. Include failures and dispositions. Saved OOF predictions and
  evaluation scripts support later comparisons without retraining when affordable.
- Current deployment/active-work consumers and a concrete reason for each keep.
  Historical references and former selection are context, not live dependencies.
  Checkpoints outside the justified keep/hold set do not need an individual
  historical rejection to become removal candidates.
- Preserve the actual config bytes and effective settings (including inherited
  defaults and CLI/wrapper overrides). Record which recipe inputs are verified;
  do not substitute current defaults silently. A matching registered config hash
  plus saved resolved parameters is useful evidence of the historical recipe.
- Reproduction command and inputs, missing prerequisites, approximate compute/time
  cost if supported by observed run history, and expected fidelity. Do not launch
  retraining to validate a proposal. Distinguish retaining results, rerunning a
  recipe, recovering identical weights, and resuming the exact old trajectory.

Preserve existing records by reference where already immutable and accessible;
do not duplicate large data/checkpoints just to create a cleanup report. If small
missing records can be assembled from existing evidence, save a new reproduction
record outside candidate paths before recommending removal. Mark unverifiable
inference as such. Credential-redact commands/configs while retaining required
environment variable names and a secure reference to their authorized source.

## Human-readable proposal

Identify proposal ID/version, timestamp, repository revision/dirty state,
inventory scope, storage roots/backends and the deployment-plan evidence used.
Summarize covered versus uninspected experiments and why. Separate evidence from
recommendations and from approvals (initially none).

Lead with a short **keep list**: current final-model members and dependencies,
explicit continuation/rollback needs, and a few justified reference models.
Separate checkpoint retention from retaining the experiment's metadata. Describe
why each chosen weight variant is needed and which duplicates/states are not.

Then show current storage and proposed remaining storage, broken down into kept
weights, held weights and metadata. Explain every substantial retained category.
The main removal proposal should cover all eligible checkpoints outside the keep
set, not merely duplicates or formally rejected experiments.

Use a table with one artifact/action per row:

| ID | Run/fold | Exact path/object version | Role | Allocated/logical bytes | Decision | Evidence/rationale | Recovery | Consumers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

Include all keep/hold items in the manifest even if the human table summarizes
large groups. Show estimated reclaimable bytes by filesystem/backend and action
set, account for shared links/replicas, and state unknown reclamation. Rank
candidates by storage savings and practical consequences, with no automatic score
cutoff. Assume exploratory runs are unlikely to be rerun. Scientific history is
usually preserved through metadata, not a checkpoint for every experiment. For each run link the reproduction record and retained
metadata. Explain higher-risk alternatives separately, such as keeping inference
weights but sacrificing resumability; such transformations also require approval.

End with the exact recommended action IDs and a request for approval of that
versioned set. If blockers prevent a defensible set, say so rather than requesting
blanket cleanup authority. No executable deletion script is part of a proposal.

## Machine-readable manifest

Write a new JSON file for each version; do not overwrite an approved manifest.
Use project-appropriate output locations outside experiment artifact directories.
This is a field contract, not a requirement to introduce a new registry framework:

```json
{
  "schema_version": 1,
  "proposal_id": "cleanup-YYYYMMDD-v1",
  "created_at": "UTC timestamp",
  "repository": {"revision": "commit", "dirty_evidence": "reference or null"},
  "scope": {"roots": [], "backends": [], "exclusions": [], "unavailable": []},
  "deployment_evidence": [],
  "experiments": [],
  "artifacts": [],
  "keep_action_ids": [],
  "hold_questions": [],
  "recommended_action_ids": [],
  "storage_before_bytes": null,
  "storage_after_bytes": {"kept_weights": null, "held_weights": null, "metadata": null},
  "optional_action_sets": [],
  "estimated_reclaimable_bytes_by_location": {},
  "reclamation_limits": [],
  "approval": null
}
```

Each experiment includes ID/status, results with evaluation contract, specifics,
reproduction-record references, recovery fidelity/gaps and retained metadata.
Each artifact includes unique action ID, experiment ID, exact path/URI and remote
version when available, root/backend, role, logical/allocated bytes, observed
mtime, local device/inode/link count or remote version identity, checksum if known,
aliases/duplicate group, keep/hold/propose-remove decision, action, evidence,
consumers, and reason. Each keep must name its current purpose; each hold must
name the unresolved question and resolution condition. Removal rationale may be
"outside the current final/reference keep set"; individual inferiority is not
required. Distinguish reproducibility limitations that need explicit acceptance
from actual live dependencies that block removal. Use null for unknown bytes/hashes, not zero or fabricated
checksums. Do not count duplicate aliases as independently reclaimable.

Hash the final manifest and show the hash with the approval request. Record later
approval in a separate receipt referencing this hash; leave the proposal immutable.
Receipts contain the exact approved IDs/actions, user approval reference, preflight
observations, per-item outcomes/timestamps and measured storage changes. New
artifacts at old paths, new consumers, or changed reproduction assumptions invalidate
the affected actions until explicitly approved again.
