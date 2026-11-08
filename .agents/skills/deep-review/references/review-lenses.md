# Select relevant review lenses

These are prompts for testing contracts, not a mandatory cross-product of tests.
Record which matters for this change and which is outside the verified scope.

## Lifecycle and state

- Initialization versus restoration; fit/resume versus validate/test/predict.
- Hook order from installed framework source or a real bounded integration run.
- Raw, transformed, cached and externally consumed state; ownership and aliases.
- Save before/during/after temporary state installation; restore on failure.
- Resume versus uninterrupted execution using identical initial inputs/state.
- Old serialized artifacts under new defaults; explicit overrides versus inference.

A useful oracle: give raw and inference weights visibly different values. A test
where they are equal cannot detect loading the wrong representation. Likewise,
tied parameters need a nonzero decay and more than one alias to reveal repeated
updates. Preserve required historical behavior rather than claiming compatibility
because deserialization did not raise.

## Counts, reductions and mapping

- Rows versus sampled examples, local versus global counts, batch versus update.
- Padding, dropping, limits, accumulation, partial final batches and zero counts.
- Mean of means versus summed numerators/denominators; weighted/masked reductions.
- NaN/Inf and invalid values before mapping; missing is not automatically negative.
- Undefined output versus a fabricated valid-looking default.

Use unequal batches/weights and values near mapping boundaries. An equal-size
example can pass both the correct and incorrect reduction. A schedule test should
compare against actual loader/optimizer behavior when those are the changed
contracts, rather than reproducing the schedule's arithmetic in the assertion.

## Concurrency, errors and side effects

- Who participates in each collective, lock, transaction or callback?
- Empty/uneven workers, failed writers and branches that only one worker enters.
- Error propagation before subsequent synchronization; bounded timeout probes.
- Atomic publication versus partially visible artifacts; cleanup after failure.
- Retry/idempotency, cancellation and resource lifetime where supported.

A successful two-worker run is not evidence for writer-failure behavior. Inject a
failure at the boundary and check the outcome of every affected participant.
Use temporary files and fake endpoints rather than live external writes.

## Interfaces and provenance

- Actual launch wrappers and artifact consumers, not only the nominal core module.
- CLI defaults/types, config overrides, optional dependencies and packaging.
- Stable identifiers, coverage/uniqueness, target order, schema and version identity.
- Immutable versus mutable references; bytes needed to reproduce dirty source.
- Effective runtime settings rather than requested settings; environment overrides.

## Scope and evidence discipline

- Does the finding apply to the final reviewed revision and a supported use case?
- Does the changed contract cause it, expose it, or merely coexist with it?
- Does the reproducer distinguish a bug from a different documented policy?
- Did a repair invalidate earlier tests, coverage claims or another consumer?
- Are GPU, distributed backend, precision, platform or dependency limits explicit?
- Is an optional design improvement being mislabeled as a blocking defect?
