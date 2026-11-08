# Review contract and regression scenarios

## Training correctness

Reconstruct sample counts from the selected sampler, not table rows. Account for
DDP padding, drop_last, integer/fractional batch limits, partial accumulation,
max_steps, scheduler interval and explicit horizon overrides. Trace scheduler
construction and stepping to the same interval. Compare optimizer updates to EMA
updates through accumulation and resume. Verify first/warmup-boundary decays,
ordinary versus Switch EMA, BatchNorm buffers and integer counters. Save while
EMA is installed and after restoration: inference must match validation and
resume must recover raw parameters plus optimizer/scheduler/EMA counters.

Inspect encoder chunking as a normalization parameter. Check compatible forward
collective counts across ranks, and numerical inference parity within declared
precision tolerances. Do not infer bitwise equivalence.

Primary implementation references are the pinned installed PyTorch optimizer,
DistributedSampler and DDP source, Lightning loop/checkpoint connector source,
and the tested template code. Consult official documentation when needed; record
the actual versions. Never rely solely on remembered hook ordering.

## Distributed evaluation

Exercise N % world_size != 0 and N < world_size, not only balanced shards. Check
forward collectives, empty loaders, epoch-end collectives, metric state shape,
and equal callback order. Reconstruct metrics from saved stable IDs and target
order; reject missing or duplicate IDs. Loss reductions require numerator and
actual denominator, including unequal batches, masks, weights and all-masked
batches. Map continuous targets before diversity tests. Undefined labels are
explicit, not measured chance-level scores. Separate group aggregation, pooled
OOF and mean fold metrics.

## Provenance and compatibility

Commit + dirty=true cannot reconstruct source. Preserve dirty file bytes,
deletions, nonignored untracked files and relevant launch wrappers in an
immutable source artifact; never archive secrets. Bind data/label/split versions
to immutable manifests and hashes. Prompt provenance is optional, but generated
outputs need immutable prompt versions, rendering inputs, model revision,
decoding parameters and output hashes when used. Check old checkpoint/config
compatibility explicitly; changing defaults must not silently rewrite resumes.

## Migration criteria

Confirmed bugs need bounded reproducers and regression tests. Generalize names,
field contracts and dependency boundaries for reusable improvements. Keep private
paths, clinical/competition mappings, submission fallback policy and unproven
architectures downstream. Optional architectures require shape, pretrained load,
precision and inference tests before promotion. Record expensive checks not run.

## Skill scenarios

1. Dirty nested config plus untracked wrapper: inventory both; discover fixes in
   the wrapper even when the nominal core file is unchanged.
2. Identical upstream implementation: classify already upstream; do not duplicate.
3. Project-only label routing: retain downstream despite successful experiments.
4. Missing optional dependency: report unverified interface; no installation.
5. All tests green but uneven rank shards untested: reproduce or label hypothesis.
6. Credentials in remotes/.env: redact remote credentials and exclude secret files.
