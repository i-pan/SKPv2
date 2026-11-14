# Reusable component review — 2026-11-14

Scope: ten downstream-derived additions described in
[REUSABLE_SPATIAL_AUXILIARY.md](REUSABLE_SPATIAL_AUXILIARY.md), based on SKPv2
`55779587dd0aeb0a2c1ecbe2f2268b503262f880`. Work was isolated from pre-existing
queue edits in another worktree. No project data or training runs were altered.

Deep-review included producer/consumer tracing, failure and compatibility checks,
and a fresh-context read-only reviewer. No remaining findings within the
reviewed scope. The reviewer confirmed the fix for FP16 weight-sum overflow;
a 512 × 512 regression checks the expected BCE, scale invariance, and gradients.
Encoder-prefix loading also preserves the original state dictionary when trying
the alternate prefix, fixing an existing fallback exposed by precedence tests.

Evidence on the final source:

- Full CPU suite: **234 passed**, 18 warnings, 22.84 seconds.
- Ruff (`src tests`) and `git diff --check`: passed.
- Real timm ResNet18 auxiliary-model forward/backward and classifier-only parity.
- Independent positive/negative cycle exhaustion, rewind/replay, DDP padding,
  zero/one sampling probabilities and scheduler sizing.
- Tiny Lightning fit/resume: rows match deterministic replay; final global step
  and cosine scheduler step both equal the intended four-update horizon.
- Positive-only sample/batch Dice oracles, empty selections, weighted BCE,
  and real BaseTask mixed-denominator validation aggregation.
- CPU one- and two-rank best/final exports verified against checkpoint and archive
  hashes; changed checkpoint, corrupt archive and missing sidecar fail closed.
- Shared segmentation normalization, frozen BN buffers, and component precedence.
- Holdout/ancestor overlap checks and independent validation worker overrides.

The existing environment was reused without installation. CUDA was hidden for
CPU distributed tests. An initial full-suite invocation omitted explicit
PYTHONPATH, causing two subprocesses to import the installed downstream checkout;
the final invocation explicitly inherited this worktree's source path and passed.

Limits: no GPU training, medical-data evaluation, or downstream checkpoint
migration was performed. The generic model was exercised with ResNet18, not all
timm architectures. WebDataset resource selection was checked without an actual
WebDataset training pipeline. Full ancestry remains caller-supplied. Auxiliary
batch Dice is local to each rank, and its validation average is not volume Dice.

## Main integration verification

Before publication, the component commit was rebased onto main
`010eff3b81ce9368583e201c009c16728744df45` to exclude two unrelated storage/setup
commits. The only conflict was the appended decision-log entry; main's existing
history and the new component entry were retained. Component source and tests
are unchanged. The full suite on this main-based tree passed: **186 passed**,
18 warnings, 13.35 seconds; Ruff and diff whitespace checks also passed. The
smaller suite excludes tests from those separate storage/setup commits.
