from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DOCS = {
    "experiment-history": ROOT / "docs" / "EXPERIMENT_HISTORY.md",
    "decision-log": ROOT / "docs" / "DECISION_LOG.md",
}


def test_required_project_documentation_exists() -> None:
    for marker, path in PROJECT_DOCS.items():
        assert path.is_file(), f"Required project documentation is missing: {path}"
        assert (
            f"skpv2-project-documentation: {marker}" in path.read_text()
        ), f"{path} must retain its SKPv2 documentation-contract marker"


def test_project_decision_log_is_distinct_from_template_decisions() -> None:
    decision_log = PROJECT_DOCS["decision-log"].read_text()
    assert "DECISIONS.md" in decision_log
    assert "template/framework" in decision_log
    assert "EXPERIMENT_HISTORY.md" in decision_log


def test_experiment_history_links_to_detailed_decisions() -> None:
    experiment_history = PROJECT_DOCS["experiment-history"].read_text()
    assert "DECISION_LOG.md" in experiment_history
    assert "DECISIONS.md" in experiment_history


def test_agent_contract_covers_bootstrap_tracking_and_documentation() -> None:
    agent_contract = (ROOT / "AGENTS.md").read_text()
    required_contracts = (
        "git clone <skpv2-template-url> .",
        "git remote rename origin template",
        "uv sync --group train --group dev",
        "Mandatory orientation",
        "DECISION_LOG.md",
        "EXPERIMENT_HISTORY.md",
        "MLflow is fail-closed",
        "Never add `--allow_local_mlflow` on your own",
    )
    for contract in required_contracts:
        assert contract in agent_contract
