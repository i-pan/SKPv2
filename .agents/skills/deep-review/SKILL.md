---
name: deep-review
description: Perform a thorough implementation or pre-commit code review within one request, using contract tracing, adversarial verification, a fresh-context reviewer, and a final verification gate. Use when the user asks for a deep, comprehensive, final, or readiness review. Not a promise of finding every defect or a substitute for a domain-specific audit.
---

# Deep review

One review request includes multiple distinct passes and verification of any
in-scope fixes. Do not require the user to repeatedly ask “review again” to reach
ordinary integration and failure-path checks. Do not promise exhaustive bug
removal: give a scoped readiness assessment backed by evidence.

## Scope and authority

- Identify the intended repository, base revision, branch, status and complete
  applicable instructions. Review staged, unstaged and relevant untracked source
  when scope is uncommitted work; a staged-only request reviews the index bytes.
- Preserve existing user edits and staging boundaries. Record which version was
  reviewed. If another actor changes it, invalidate affected conclusions.
- Review-only requests produce findings without modifying source or staging.
  Existing implementation/fix authorization remains valid: when review is part
  of finishing that work, fix confirmed in-scope defects without re-asking.
  Prepare/stage/commit/push only to the extent separately authorized.
- Match depth to consequence and change surface. Do not add architectures,
  features, broad refactors or unrelated hardening to “complete” a review.

## 1. Derive contracts before reading the implementation as the answer

Read the request, relevant documentation and callers. Build a compact internal
coverage table: contract → entry point/consumer → observable behavior → evidence
or open gap. Separate required behavior from assumptions and optional advice.
Use [review-lenses.md](references/review-lenses.md) for relevant boundary cases;
do not run every domain checklist on every change.

Inspect full changed functions and their reachable producers/consumers, including
unchanged call sites, CLI paths, wrappers, alternate modes, serialization readers
and packaging. Include changed defaults and historical compatibility. Passing a
function unit test does not verify how its framework invokes it.

## 2. Review through distinct lenses

First trace the ordinary end-to-end operation against the contracts. Then try to
break those contracts at state transitions, invalid/empty inputs, alternate modes,
partial completion and component boundaries. Check lifetime/order, aliasing,
units/counts, exceptional control flow and distributed/concurrent participation
where applicable. Inspect relevant failures and workarounds, not only green tests.

For each suspected defect, identify a concrete trigger, expected result, actual
result and user-visible consequence. Establish it with a bounded reproducer or a
complete source trace. Label untested hypotheses explicitly. Never invent a bug
to satisfy an assumed review quota, and do not report preference as correctness.
Compare the base where useful: distinguish regressions, pre-existing defects
exposed by this change, and unrelated existing issues.

## 3. Verify behavior, not agreement with the implementation

Tests should derive expected results from the contract or an independent oracle.
Prefer discriminating examples: unequal sizes, different raw/inference weights,
nontrivial resume counters, missing participants, and invalid values that could
otherwise produce plausible success. Test public entry points or real lifecycle
hooks for integration-sensitive changes. A mock of the behavior being questioned
cannot certify that behavior.

For confirmed defects being fixed, make the reproducer fail before and pass after
when practical. If executing the old behavior is unsafe or unavailable, record
that limitation instead. Keep new tests proportional to risk; do not add tests
that merely restate a trivial implementation. Run bounded local checks using
existing dependencies. Do not install packages, launch training, spend cloud
resources or run expensive GPU checks solely because a review suggests them.

## 4. Obtain a fresh-context challenge

For nontrivial changes, use a read-only review sub-agent when available and allowed
by higher-priority instructions. Give it the scope/revisions, original requirements,
applicable instructions and repository access, but omit the implementer's proposed
findings, claimed success and preferred fix. Ask it to trace consumers and identify
concrete bugs or missing verification, allowing “no findings.” Instruct it not to
edit files or delegate further. Do not launch competing source edits.

If independent review is unavailable or prohibited, perform a separate consumer-
and-failure-path pass and disclose that it was self-review, not independent review.
Do not silently change models or invoke external services to simulate independence.
Adjudicate findings with evidence; reviewer agreement is not proof and disagreement
is not automatically a defect. Independent review can run alongside read-only
inspection, but it must assess the final relevant code after fixes.

## 5. Close the review internally

When fixes are authorized, resolve confirmed in-scope defects, inspect their
knock-on effects and rerun affected checks. Have the fresh reviewer check changed
contracts again when fixes invalidate its assessment. Do not resend the same
unchanged code for repeated speculative reviews. After focused checks pass, run
required repository checks and the full suite when practical on the final code.
Do not broaden or repeat testing after that gate unless a new change, failure or
unresolved concern justifies it.

Readiness requires all of the following:

- Required contracts and affected entry points have evidence, or explicit gaps.
- Every confirmed material in-scope defect is fixed and verified, or reported open.
- Relevant failure/alternate-mode paths have been checked, not just the happy path.
- A final distinct review pass has no unaddressed material findings.
- Tests/lint/other required checks apply to the exact final code being delivered.
- Documentation and compatibility claims match the verified behavior.

If repeated passes keep producing material defects in the same subsystem, stop
patching symptoms: revisit that subsystem's contract/design and test strategy
within the authorized scope. Honor user budgets. Unavailable environments or
unresolved requirements mean an explicit conditional/incomplete assessment, not
“all clear.” Never wait indefinitely for certainty or claim that a finite review
proves absence of defects.

## Report

Lead with readiness or prioritized unresolved findings. Summarize important fixes,
verification, reviewed scope, independent versus self-review, and concrete limits.
State whether changes were staged, committed, pushed or launched. Avoid dumping the
internal checklist; link a review record only when useful or requested. Use “no
remaining findings within the reviewed scope,” never “bug-free” or “everything is
covered.” Do not require a follow-up “review again” for checks already in scope.
