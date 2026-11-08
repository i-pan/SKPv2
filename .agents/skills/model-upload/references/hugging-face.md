# Hugging Face publication

The SKILL.md proposal gate applies before this procedure. Read-only inspection
can inform a proposal; repository creation, packaging, and upload require explicit
approval of the corresponding actions in the presented proposal. Confirmed HF
account/names/visibility alone are not execution approval.

1. Inspect installed `huggingface_hub`/CLI and current official upload guidance.
   Use an existing framework integration when it preserves the release contract;
   otherwise use the official Hub API. No hard-coded account, repo, or token.
2. Verify authenticated identity and target permissions without printing secrets.
   Confirm repository ID, model repo type, visibility, and whether it is new or
   existing. For an existing repo, inspect contents and latest revision. Never
   silently turn private content public, delete stale files, or overwrite a
   release owned by another workflow. Explicit authorization remains necessary.
3. Add valid YAML model-card metadata: appropriate task/library/tags and verified
   license/dataset/base-model fields. Use actual Hub IDs where the schema calls
   for them; put non-Hub dataset links in prose. Do not advertise Transformers or
   hosted inference support for a custom runtime that does not implement it.
   Verify that source/reproduction links are prominent near the top and that
   the model-specific training procedure precedes inference examples, as defined
   in the shared release contract; do not bury them in provenance boilerplate.
   Link companion models and public source. Local-only cards need not claim a
   public license or public source repository.
4. Upload only the validated inventory. Use `upload_folder`/CLI for normal uploads;
   consider `upload_large_folder` for large/resumable releases, after checking its
   installed semantics. Do not presume one atomic commit: resumable uploads can
   expose partial content. Use a draft branch/private staging only with agreed
   policy, or clearly label candidates until complete. Preserve resume cache and
   do not run competing uploaders on the same folder. Stop on authentication,
   rights, quota, or hash failures; use bounded retries for transient transport
   failures and report partial remote state rather than retry forever.
5. Track every created repository and remote revision. Verify actual visibility
   and file coverage against the allowlist. Check remote content hashes (download
   pinned artifacts when metadata is insufficient; do not assume an ETag is a
   SHA-256). Smoke-load from the pinned remote revision in an isolated cache when
   feasible. Report skipped runtime checks honestly.
6. Only mark/tag a validated release after the full package is available. Record
   a delivery receipt with repo ID, immutable commit, inventory hash, validation,
   and retrieval example. Do not silently tag, push source, or publish additional
   artifacts outside the authorized release.

Document both a full-package retrieval example and a single-checkpoint download.
For ensembles, avoid forcing all weights to download for a single-checkpoint
use case; expose `hf_hub_download` or a targeted snapshot plus config/runtime.
Explain whether presets reduce execution only or download/storage as well.
