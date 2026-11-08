# Official guidance and provenance

Consulted 2026-11-08. Recheck mutable provider interfaces, limits, and metadata
schemas at execution time. This skill synthesizes guidance; it does not certify
compliance with any company's internal release standard.

- Hugging Face, [Model Cards](https://huggingface.co/docs/hub/model-cards):
  README-based documentation and structured metadata for discoverability.
- Hugging Face, [Upload files](https://huggingface.co/docs/huggingface_hub/en/guides/upload):
  official upload APIs and resumable transfer options; use installed-version semantics.
- Hugging Face, [Uploading models](https://huggingface.co/docs/hub/en/models-uploading):
  library integrations and custom model publication.
- Hugging Face, [Pickle Scanning](https://huggingface.co/docs/hub/security-pickle):
  serialization security considerations; scanners are not a safety guarantee.
- Google Research, [Model Cards for Model Reporting](https://research.google/pubs/model-cards-for-model-reporting/):
  intended use, evaluation context, subgroup performance, limitations, and ethical
  considerations motivate the shared documentation contract.
- Microsoft, [What is responsible AI?](https://support.microsoft.com/en-us/privacy/what-is-responsible-ai):
  transparency, accountability, fairness, reliability/safety, privacy/security,
  and inclusiveness inform risk and limitation reporting.

Local versioned folders, approval boundaries, exact payload inventories,
checkpoint-state provenance, and post-transfer verification are workflow choices
in this skill. Do not attribute these exact mechanics to the sources above.
