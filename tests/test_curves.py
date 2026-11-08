from types import SimpleNamespace as NS

from skp.curves import (
    build_parser,
    latest_by_step,
    load_curves,
    render_panel,
    resolve_runs,
    validation_eta,
    render_latest_comparison,
)


def point(step, value, timestamp=0):
    return NS(step=step, value=value, timestamp=timestamp)


def run(name="cfg/exp-0363", fold="0", run_id="a" * 32):
    return NS(
        info=NS(run_id=run_id, start_time=0, status="RUNNING"),
        data=NS(
            tags={"mlflow.runName": name},
            params={"fold": fold},
            metrics={"val_metric": 0.8, "val/cohort/loss": 0.2},
        ),
    )


class Client:
    def get_metric_history(self, run_id, key):
        return {
            "epoch": [point(100, 0), point(200, 1)],
            "val_metric": [point(100, 0.7), point(200, 0.8)],
            "val/cohort/loss": [point(100, 0.4), point(200, 0.2)],
        }.get(key, [])


def test_parser_multiple_metrics():
    args = build_parser().parse_args(
        [
            "--experiment-name",
            "test",
            "exp-0348",
            "exp-0363",
            "-m",
            "val_metric",
            "cohort/loss",
            "-m",
            "all/loss",
        ]
    )
    assert args.metric == ["val_metric", "cohort/loss", "all/loss"]
    assert (
        build_parser().parse_args(["--experiment-name", "test", "exp-0363"]).metric
        is None
    )


def test_epoch_alignment_aliases_missing_metrics_and_zoom():
    panels, warnings = load_curves(
        Client(), [run()], ["val_metric", "cohort/loss", "bad"]
    )
    assert panels["val_metric"][0][2] == [(1, 0.7), (2, 0.8)]
    assert panels["cohort/loss"][0][2] == [(1, 0.4), (2, 0.2)]
    assert len(warnings) == 1
    panels, _ = load_curves(Client(), [run()], ["val_metric"], start=2)
    assert panels["val_metric"][0][2] == [(2, 0.8)]


def test_missing_epoch_is_not_silently_invented():
    class NoEpoch(Client):
        def get_metric_history(self, run_id, key):
            return [] if key == "epoch" else super().get_metric_history(run_id, key)

    panels, warnings = load_curves(NoEpoch(), [run()], ["val_metric"])
    assert not panels["val_metric"]
    assert "--x-axis step" in warnings[0]
    panels, _ = load_curves(NoEpoch(), [run()], ["val_metric"], "step")
    assert panels["val_metric"][0][2][0] == (100, 0.7)


def test_latest_duplicate_step():
    assert latest_by_step([point(1, 0.9, 2), point(1, 0.1, 1)]) == {1: 0.9}


def test_half_epoch_positions_comparison_and_eta():
    r = run()
    r.data.params.update(
        {
            "args/val_check_interval": "0.5",
            "n_train": "3472",
            "world_size": "2",
            "batch_size": "4",
            "num_epochs": "3",
        }
    )
    histories = {
        "epoch": [point(216, 0), point(433, 0), point(650, 1)],
        "val_metric": [
            point(216, 0.8, 100000),
            point(433, 0.9, 200000),
            point(650, 0.91, 300000),
        ],
    }
    client = NS(get_metric_history=lambda _, key: histories.get(key, []))
    panels, warnings = load_curves(client, [r], ["val_metric"])
    assert not warnings
    assert panels["val_metric"][0][2] == [(0.5, 0.8), (1, 0.9), (1.5, 0.91)]
    text = validation_eta(client, r, now=350)
    assert "in 0m 50s" in text and "[2 interval(s)]" in text
    curves = panels["val_metric"] + [(1, "full epoch", [(1, 0.85)])]
    text = render_latest_comparison("auc", curves, color=False)
    assert "LATEST MATCHED EPOCH 1" in text and "0.90000" in text
    rendered = render_panel("auc", curves, color=False)
    assert "0.5" in rendered and "1.5" in rendered
    # A partial first epoch is not mistaken for a completed first epoch.
    histories["epoch"] = histories["epoch"][:1]
    histories["val_metric"] = histories["val_metric"][:1]
    assert "first validation interval estimate" in validation_eta(client, r, now=150)
    # Missing metadata is fail-closed, never duplicate x=1 for two validations.
    del r.data.params["world_size"]
    panels, warnings = load_curves(client, [r], ["val_metric"])
    assert not panels["val_metric"] and warnings


def test_renderer_constant_single_point_and_color():
    curves = [(0, "run A", [(0, 0.8)])]
    text = render_panel("val_metric", curves, color=False)
    assert "run A" in text and "max 0.8000 @ 0" in text
    assert any(c in text for c in "▀▄█")
    assert "\033" not in text
    assert "\033[38;5;" in render_panel("val_metric", curves, color=True)
    assert "no data" in render_panel("missing", [])


def test_run_resolution_exact_match_pagination_folds_and_deduplication():
    class Page(list):
        token = None

    class SearchClient:
        def get_experiment_by_name(self, name):
            return NS(experiment_id="1")

        def search_runs(self, ids, max_results, page_token):
            if page_token is None:
                page = Page([run(), run("cfg/exp-03630")])
                page.token = "next"
                return page
            return Page([run(fold="1", run_id="b" * 32)])

    assert len(resolve_runs(SearchClient(), ["exp-0363"], "test")) == 2
    matches = resolve_runs(SearchClient(), ["exp-0363", "exp-0363"], "test", fold=0)
    assert len(matches) == 1


def test_eta_wall_clock_cadence_and_overdue():
    class TimedClient:
        def get_metric_history(self, run_id, key):
            if key == "validation/epoch_completed":
                return []
            if key == "epoch":
                return [point(i, i) for i in range(4)]
            return [point(i, 0.8, t * 1000) for i, t in enumerate([100, 200, 310, 400])]

    text = validation_eta(TimedClient(), run(), now=450)
    assert "in 0m 50s" in text
    assert "cadence 1m 40s [3 interval(s)]" in text
    assert "overdue by 0m 20s" in validation_eta(TimedClient(), run(), now=520)


def test_eta_finished_and_final_validation():
    r = run()
    r.info.status = "FINISHED"
    assert "no next validation" in validation_eta(Client(), r, now=1)
    r.info.status = "RUNNING"
    r.data.params["num_epochs"] = "2"
    assert "final validation logged" in validation_eta(Client(), r, now=1)


def test_eta_first_epoch_bootstrap_then_observed_interval():
    histories = {
        "epoch": [point(100, 0)],
        "val_metric": [point(100, 0.8, 160000)],
    }
    client = NS(get_metric_history=lambda _, key: histories.get(key, []))
    r = run()
    r.info.start_time = 10000
    text = validation_eta(client, r, now=200)
    assert "in 1m 50s" in text
    assert "first epoch estimate 2m 30s [includes startup]" in text
    assert "overdue by 0m 10s" in validation_eta(client, r, now=320)
    histories["epoch"].append(point(200, 1))
    histories["val_metric"].append(point(200, 0.9, 260000))
    text = validation_eta(client, r, now=300)
    assert "in 1m 00s" in text
    assert "observed cadence 1m 40s [1 interval(s)]" in text
    assert "first epoch estimate" not in text


def test_eta_first_result_missing_start_or_resumed_epoch():
    histories = {"epoch": [point(100, 0)], "val_metric": [point(100, 0.8, 160000)]}
    client = NS(get_metric_history=lambda _, key: histories.get(key, []))
    r = run()
    for start in [None, 160000, 200000]:
        r.info.start_time = start
        assert "warming up" in validation_eta(client, r, now=200)
    r.info.start_time = 10000
    histories["epoch"] = [point(100, 5)]
    assert "warming up" in validation_eta(client, r, now=200)


def test_eta_missing_data_and_epoch_gaps():
    r = run()
    r.data.metrics = {}
    assert "awaiting validation" in validation_eta(Client(), r)

    class GapClient:
        def get_metric_history(self, run_id, key):
            if key == "validation/epoch_completed":
                return []
            if key == "epoch":
                return [point(1, 0), point(2, 2)]
            return [point(1, 0.8, 100000), point(2, 0.9, 300000)]

    assert "warming up" in validation_eta(GapClient(), run(), now=350)


def test_epoch_grid_ticks_and_labels():
    text = render_panel(
        "auc",
        [(0, "run", [(i, 0.8 + 0.001 * i) for i in range(15)])],
        width=100,
        color=False,
    )
    assert "┊" in text and "┬" in text
    lines = text.splitlines()
    axis = next(i for i, line in enumerate(lines) if "└" in line)
    assert lines[axis + 1].split() == [str(i) for i in range(15)]
    assert "epoch (completed)" in lines[axis + 2]
    assert "\033" not in text


def test_latest_comparison_matches_epochs_not_last_scores():
    curves = [
        (0, "baseline", [(0, 0.7), (1, 0.8), (2, 0.99)]),
        (1, "active", [(0, 0.6), (1, 0.81)]),
    ]
    text = render_latest_comparison("auc", curves, color=False)
    assert "LATEST MATCHED EPOCH 1" in text
    assert "0.80000" in text and "0.81000" in text and "+0.01000" in text
    assert "0.99000" not in text
    assert "later results through epoch 2" in text
    assert "\033" not in text


def test_latest_comparison_no_interpolation():
    curves = [(0, "A", [(0, 0.7), (2, 0.8)]), (1, "B", [(1, 0.75)])]
    assert "no shared epoch" in render_latest_comparison("auc", curves)


def test_collisions_are_side_by_side_including_boundaries():
    curves = [(0, "A", [(0, 0.8), (1, 0.9)]), (1, "B", [(0, 0.8), (1, 0.9)])]
    text = render_panel("auc", curves, color=True)
    assert "collisions offset horizontally" in text
    import re

    colored = re.findall(r"\x1b\[38;5;208m[▀▄]\x1b\[0m", text)
    assert len(colored) == 2
    plain = re.sub(r"\x1b\[[0-9;]*m", "", text)
    chart_rows = [line for line in plain.splitlines() if "│" in line]
    assert sum(sum(line.count(c) for c in "▀▄█") for line in chart_rows) == 4
    assert sum(bool(re.search("[▀▄█]{2}", line)) for line in chart_rows) == 2


def test_adjacent_cells_keep_distinct_colors():
    import re

    # Equal x, nearby y values land in adjacent terminal rows, not one cell.
    curves = [
        (0, "A", [(0, 0.40), (1, 0.40)]),
        (1, "B", [(0, 0.60), (1, 0.60)]),
        (2, "bounds", [(2, 0), (3, 1)]),
    ]
    text = render_panel("auc", curves, height=6, color=True)
    for code in (39, 208):
        assert re.search(rf"\x1b\[38;5;{code}m[▀▄]\x1b\[0m", text)


def test_blocks_only_at_logged_points():
    text = render_panel("auc", [(0, "A", [(0, 0.7), (10, 0.9)])], color=False)
    chart_rows = [line for line in text.splitlines() if "│" in line]
    assert sum(sum(line.count(c) for c in "▀▄█") for line in chart_rows) == 2


def test_half_block_independent_colors_and_empty_halves():
    from skp.curves import half_block

    assert half_block(0, 1, True) == "\033[38;5;39;48;5;208m▀\033[0m"
    assert half_block(0, None, True) == "\033[38;5;39m▀\033[0m"
    assert half_block(None, 1, True) == "\033[38;5;208m▄\033[0m"
    assert half_block(0, 1, False) == "█"
    assert half_block(None, None, True) == " "


def test_explicit_fractional_epochs_support_custom_samplers():
    from skp.curves import completed_epochs

    model_run = run()
    model_run.data.params["sampler"] = "GroupBalancedSampler"
    client = NS(
        get_metric_history=lambda _, key: (
            [point(3, 0.5), point(7, 1.0)]
            if key == "validation/epoch_completed"
            else []
        )
    )
    assert completed_epochs(client, model_run) == {3: 0.5, 7: 1.0}


def test_odd_batch_fractional_checks_follow_lightning_floor_and_epoch_wrap():
    from skp.curves import completed_epochs

    r = run()
    r.data.params.update(
        {
            "args/val_check_interval": "0.5",
            "n_train": "10",
            "world_size": "1",
            "batch_size": "2",
        }
    )
    histories = {
        "epoch": [point(1, 0), point(3, 0), point(6, 1)],
        "val_metric": [
            point(1, 0.7, 100000),
            point(3, 0.8, 200000),
            point(6, 0.9, 350000),
        ],
    }
    client = NS(get_metric_history=lambda _, key: histories.get(key, []))
    assert completed_epochs(client, r) == {1: 0.4, 3: 0.8, 6: 1.4}
    panels, warnings = load_curves(client, [r], ["val_metric"])
    assert not warnings
    assert panels["val_metric"][0][2] == [(0.4, 0.7), (0.8, 0.8), (1.4, 0.9)]
    # The next .4-epoch span uses the prior .4 span, not the .6 epoch-wrap span.
    assert "in 0m 50s" in validation_eta(client, r, now=400)
    histories["validation/epoch_completed"] = [
        point(1, 0.4),
        point(3, 0.8),
        point(6, 1.4),
    ]
    assert "in 0m 50s" in validation_eta(client, r, now=400)
    # Before an epoch-wrap duration is observed, refuse to invent it.
    histories["val_metric"] = histories["val_metric"][:2]
    assert "warming up" in validation_eta(client, r, now=250)


def test_historical_fractional_accumulation_uses_completed_updates():
    from skp.curves import completed_epochs

    r = run()
    r.data.params.update(
        {
            "args/val_check_interval": "0.5",
            "n_train": "14",
            "world_size": "1",
            "batch_size": "2",
            "accumulate_grad_batches": "2",
        }
    )
    client = NS(
        get_metric_history=lambda _, key: (
            [point(0, 0), point(2, 0)] if key == "epoch" else []
        )
    )
    assert completed_epochs(client, r) == {0: 3 / 7, 2: 6 / 7}
    # Multiple checks before an optimizer update cannot be reconstructed uniquely.
    r.data.params["args/val_check_interval"] = "0.25"
    assert completed_epochs(client, r) == {}


def test_eta_has_no_project_specific_metric_fallback():
    r = run()
    r.data.metrics = {"val/expert/competition_macro_auc": 0.8}
    assert "awaiting validation results" in validation_eta(Client(), r)
