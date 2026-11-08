---
name: template-review
description: Audit a downstream project for reusable upstream fixes, framework drift, and template migration, including dirty source and actual launch paths. Produces evidence-backed findings and an ordered migration backlog.
---

# Template review

Audit only unless the user explicitly requests implementation. Do not edit source,
commit, push, install packages, launch training, or incur cloud costs during an
audit. Preserve dirty worktrees and redact credentials in evidence and outputs.
Do not import arbitrary configs to inspect them: imports may launch work.

Identify roles from both repositories' remotes, branch, status, top-level files,
and complete applicable instructions. Pin both HEAD revisions and inventory
tracked modifications, deletions, and nonignored untracked source. Use
`scripts/inventory.py DOWNSTREAM TEMPLATE` for a read-only hash comparison;
it inventories bytes, not migration value. Files above the default 8 MiB bound
are explicitly unverified; raise `--max-file-bytes` deliberately when needed. Ignored files and credential-like
paths are excluded explicitly; identify relevant ignored launch artifacts by
manual inspection without publishing secrets. Hashes alone do not preserve dirty
source: recommend an authorized source archive when reproducibility is needed.

Read [review.md](references/review.md) for the review contract and regression
matrix. Inspect core modules AND executable launch paths: wrappers, monkeypatches,
config overrides, shell queues, inference packaging, and checkpoint readers.
Search experiment history, decision logs, and failure records for workarounds.
Trace data → sampler → rank shards → batches → optimizer updates → scheduler/EMA
→ validation → checkpoint → inference. Check resume, precision, normalization,
dependency boundaries, and provenance at each transition.

Run bounded CPU tests/reproducers when dependencies are already available.
Request authorization before expensive GPU checks. Missing dependencies are an
explicit verification limit, never evidence that an interface works. Passing
existing tests does not validate uncovered behavior. State unread areas.

Produce Markdown findings and a JSON array using the schema in
[findings.md](references/findings.md). Classify each finding as confirmed bug,
reusable improvement, optional extension, already upstream, project-only, or
hypothesis requiring verification. Distinguish observed evidence from proposed
acceptance tests. Order migration by correctness, evaluation, developer workflow,
then optional components. Recommend separate reviewable changes; never wholesale
cherry-pick downstream history. Put framework decisions in root DECISIONS.md and
project reasoning in docs/DECISION_LOG.md. Update the template backlog only when
writing it is within the requested scope.
