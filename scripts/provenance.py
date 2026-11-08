"""Optional provenance CLI. Run with `python scripts/provenance.py --help`."""

import argparse
import json
from pathlib import Path
from skp.provenance import (
    append_event,
    record_prompt_result,
    verify_prompt_results,
    verify_manifest,
    snapshot_source,
    register_prompt,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    event = sub.add_parser("event")
    event.add_argument("registry")
    event.add_argument("record", help="Complete JSON record, including status")
    verify = sub.add_parser("verify-dataset")
    verify.add_argument("reference", help="JSON dataset ID/path/SHA-256 reference")
    source = sub.add_parser("snapshot-source")
    source.add_argument("repo")
    source.add_argument("output")
    prompt = sub.add_parser("register-prompt")
    prompt.add_argument("registry")
    prompt.add_argument("prompt_id")
    prompt.add_argument("text")
    result = sub.add_parser("record-result")
    result.add_argument("registry")
    result.add_argument("record")
    validate = sub.add_parser("verify-results")
    validate.add_argument("registry")
    args = parser.parse_args()
    if args.action == "event":
        append_event(args.registry, json.loads(Path(args.record).read_text()))
    elif args.action == "verify-dataset":
        verify_manifest(json.loads(Path(args.reference).read_text()))
    elif args.action == "snapshot-source":
        print(json.dumps(snapshot_source(args.repo, args.output), indent=2))
    elif args.action == "record-result":
        record_prompt_result(args.registry, json.loads(Path(args.record).read_text()))
    elif args.action == "verify-results":
        verify_prompt_results(args.registry)
    else:
        print(json.dumps(register_prompt(args.prompt_id, args.text, args.registry)))


if __name__ == "__main__":
    main()
