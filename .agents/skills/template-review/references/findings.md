# Finding schema

Emit a JSON array. Each record has:

- `id`: stable audit identifier;
- `classification`: confirmed bug | reusable improvement | optional extension |
  already upstream | project-only | hypothesis requiring verification;
- `severity`: critical | high | medium | low;
- `title`, `affected_behavior`;
- `source_locations`: repository, pinned revision, file, lines, dirty status;
- `evidence`: observed result, reproducer command, test result or artifact;
- `generalization_required`;
- `compatibility_impact`;
- `acceptance_tests`: concrete proposed/verified cases, explicitly distinguished;
- `verification_limits`;
- `migration_order` and `disposition`.

The Markdown report should expose the same evidence and limits without requiring
readers to inspect JSON. Hypotheses must never be presented as confirmed bugs.
