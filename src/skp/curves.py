"""Read-only, terminal-native comparison of MLflow validation curves."""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import median

COLORS = (39, 208, 82, 201, 220, 159, 196, 117)


def duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return (
        f"{hours}h {minutes:02d}m {seconds:02d}s"
        if hours
        else f"{minutes}m {seconds:02d}s"
    )


def validation_eta(client, run, now=None):
    """Estimate next completed validation from actual result timestamps.

    Use up to three most recent consecutive-epoch intervals, not training-step
    timestamps. After epoch zero only, bootstrap from run start to first validation
    (includes startup). Never move an overdue estimate forward because time passed.
    """
    now = time.time() if now is None else now
    status = run.info.status
    if status != "RUNNING":
        return f"{status} · no next validation expected"
    keys = run.data.metrics
    key = "val_metric" if "val_metric" in keys else None
    if key is None:
        return "RUNNING · next validation ETA unavailable (awaiting validation results)"
    epochs = completed_epochs(client, run)
    interval = float(run.data.params.get("args/val_check_interval", 1))
    # Deduplicate by epoch so replayed metrics cannot create fake intervals.
    points = {}
    for p in sorted(
        client.get_metric_history(run.info.run_id, key), key=lambda p: p.timestamp
    ):
        epoch = epochs.get(p.step)
        if epoch is not None and math.isfinite(epoch) and math.isfinite(p.value):
            points[epoch] = p.timestamp / 1000
    ordered = sorted(points.items())
    if not ordered:
        return (
            "RUNNING · next validation ETA unavailable (no timestamped epoch results)"
        )
    last_epoch, last_time = ordered[-1]
    last_clock = datetime.fromtimestamp(last_time).astimezone().strftime("%H:%M:%S %Z")
    prefix = f"RUNNING · last validation {last_clock} ({duration(now - last_time)} ago)"
    total = run.data.params.get("num_epochs")
    if total is not None:
        try:
            if last_epoch >= int(total):
                return prefix + " · final validation logged; finishing run"
        except (ValueError, TypeError):
            pass
    layout = _historical_validation_layout(run.data.params, interval)
    fractions = (
        [batch / layout[0] for batch in layout[1]]
        if layout is not None
        else [i * interval for i in range(1, round(1 / interval) + 1)]
        if 0 < interval <= 1
        else []
    )

    def next_validation(epoch):
        return next(
            (
                base + fraction
                for base in (math.floor(epoch), math.floor(epoch) + 1)
                for fraction in fractions
                if base + fraction > epoch + 1e-9
            ),
            None,
        )

    next_epoch = next_validation(last_epoch)
    next_gap = next_epoch - last_epoch if next_epoch is not None else None
    intervals = [
        t1 - t0
        for (e0, t0), (e1, t1) in zip(ordered, ordered[1:])
        if next_gap is not None
        and next_validation(e0) is not None
        and math.isclose(e1, next_validation(e0))
        and math.isclose(e1 - e0, next_gap)
        and t1 > t0
    ][-3:]
    if intervals:
        cadence = median(intervals)
        basis = f"observed cadence {duration(cadence)} [{len(intervals)} interval(s)]"
    else:
        start_ms = getattr(run.info, "start_time", None)
        start = start_ms / 1000 if start_ms is not None else None
        if (
            len(ordered) != 1
            or not fractions
            or not math.isclose(last_epoch, fractions[0])
            or next_gap is None
            or not math.isclose(last_epoch, next_gap)
            or start is None
            or not math.isfinite(start)
            or last_time <= start
        ):
            return (
                prefix + " · ETA warming up (need two consecutive validation results)"
            )
        cadence = last_time - start
        unit = "epoch" if interval == 1 else "validation interval"
        basis = f"first {unit} estimate {duration(cadence)} [includes startup]"
    expected = last_time + cadence
    clock = (
        datetime.fromtimestamp(expected).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    )
    remaining = expected - now
    countdown = (
        f"in {duration(remaining)}"
        if remaining >= 0
        else f"overdue by {duration(-remaining)}"
    )
    return prefix + f" · next ≈ {clock} ({countdown})" + f" · {basis}"


def latest_by_step(history):
    result = {}
    for point in sorted(history, key=lambda p: p.timestamp):
        result[point.step] = point.value
    return result


def _historical_validation_layout(params, interval):
    """Reconstruct only the standard, unlimited loader's validation schedule."""
    if interval not in (0.25, 0.5, 1):
        return None
    if params.get("sampler", "None") not in {"None", ""}:
        return None
    if params.get("args/limit_train_batches", "None") != "None":
        return None
    try:
        samples = math.ceil(int(params["n_train"]) / int(params["world_size"]))
        batch_size = int(params["batch_size"])
        drop_last = str(params.get("train_drop_last", "True")).lower()
        if drop_last not in {"true", "false"}:
            return None
        batches = (
            samples // batch_size
            if drop_last == "true"
            else math.ceil(samples / batch_size)
        )
        accumulation = int(params.get("accumulate_grad_batches", 1))
        if batches < 1 or accumulation < 1:
            return None
        check_every = max(1, int(batches * interval))
        return batches, list(range(check_every, batches + 1, check_every)), accumulation
    except (KeyError, ValueError, TypeError, ZeroDivisionError):
        return None


def completed_epochs(client, run):
    """Map logged optimizer steps to completed epochs, retaining fractional checks.

    SKP's distributed train loader pads its sampler and drops the last partial
    batch. For historical half-epoch runs use those logged, explicit settings.
    Refuse unsupported sampler/limited-loader mappings rather than invent positions.
    """
    explicit = latest_by_step(
        client.get_metric_history(run.info.run_id, "validation/epoch_completed")
    )
    if explicit:
        return explicit
    epochs = latest_by_step(client.get_metric_history(run.info.run_id, "epoch"))
    params = run.data.params
    interval = float(params.get("args/val_check_interval", 1))
    if interval == 1:
        return {s: e + 1 for s, e in epochs.items()}
    layout = _historical_validation_layout(params, interval)
    if layout is None:
        return {}
    batches, checks, accumulation = layout
    steps = math.ceil(batches / accumulation)
    positions = {}
    for batch in checks:
        # Validation between updates observes the preceding optimizer step.
        updates = (
            math.ceil(batch / accumulation)
            if batch == batches
            else batch // accumulation
        )
        if updates < 1 or updates in positions:
            return {}  # Multiple validations at one MLflow step are ambiguous.
        positions[updates] = batch / batches
    mapped = {}
    for step, epoch in epochs.items():
        local_step = step + 1 - int(epoch) * steps
        if local_step in positions:
            mapped[step] = epoch + positions[local_step]
    return mapped


def resolve_runs(client, selectors, experiment_name, fold=None):
    """Resolve exact run IDs or exact trailing project experiment IDs.

    Retries and folds remain distinct curves; never silently select the latest.
    """
    experiment = client.get_experiment_by_name(experiment_name)
    available = None
    resolved = {}
    for selector in selectors:
        if re.fullmatch(r"[a-fA-F0-9]{32}", selector):
            matches = [client.get_run(selector)]
        else:
            if experiment is None:
                raise ValueError(f"MLflow experiment not found: {experiment_name}")
            if available is None:
                available = []
                token = None
                while True:
                    page = client.search_runs(
                        [experiment.experiment_id], max_results=1000, page_token=token
                    )
                    available.extend(page)
                    token = page.token
                    if not token:
                        break
            matches = [
                run
                for run in available
                if (name := run.data.tags.get("mlflow.runName", "")) == selector
                or name.rsplit("/", 1)[-1] == selector
            ]
        matches = [
            run
            for run in matches
            if fold is None or run.data.params.get("fold") == str(fold)
        ]
        if not matches:
            raise ValueError(f"No runs match {selector!r} (fold={fold})")
        for run in sorted(matches, key=lambda r: r.info.start_time or 0):
            resolved[run.info.run_id] = run
    return list(resolved.values())


def load_curves(client, runs, metrics, x_axis="epoch", start=None, end=None):
    panels = {metric: [] for metric in metrics}
    warnings = []
    for index, run in enumerate(runs):
        run_id = run.info.run_id
        name = run.data.tags.get("mlflow.runName", run_id)
        short = name.rsplit("/", 1)[-1]
        label = f"{short} f{run.data.params.get('fold', '?')} [{run_id[:7]}]"
        epochs = completed_epochs(client, run)
        for metric in metrics:
            key = metric if metric in run.data.metrics else f"val/{metric}"
            values = (
                latest_by_step(client.get_metric_history(run_id, key))
                if key in run.data.metrics
                else None
            )
            if values is None:
                warnings.append(f"{label}: missing {metric}")
                continue
            if x_axis == "epoch" and any(step not in epochs for step in values):
                warnings.append(
                    f"{label}: {metric} lacks exact epoch mapping; use --x-axis step"
                )
                continue
            points = sorted(
                (epochs[step] if x_axis == "epoch" else step, value)
                for step, value in values.items()
                if math.isfinite(value)
                and (x_axis != "epoch" or math.isfinite(epochs[step]))
            )
            points = [
                (x, y)
                for x, y in points
                if (start is None or x >= start) and (end is None or x <= end)
            ]
            if points:
                panels[metric].append((index, label, points))
            else:
                warnings.append(f"{label}: no finite values for {metric}")
    return panels, warnings


def paint(text, index, color):
    return f"\033[38;5;{COLORS[index % len(COLORS)]}m{text}\033[0m" if color else text


def render_latest_comparison(metric, curves, color=True, x_axis="epoch"):
    """Compare exact shared observations, never different epochs or interpolation."""
    if not curves:
        return ""
    mappings = [dict(points) for _, _, points in curves]
    shared = set(mappings[0])
    for values in mappings[1:]:
        shared.intersection_update(values)
    if not shared:
        return f"{metric}: no shared {x_axis} for an exact comparison"
    latest = max(shared)
    baseline = mappings[0][latest]
    labels = [label for _, label, _ in curves]
    width = max(len(label) for label in labels)
    title = f"{metric} · LATEST MATCHED {x_axis.upper()} {latest:g}"
    if x_axis == "epoch":
        title += " (completed)"
    lines = [title, f"  {'Run':<{width}}  {'Value':>9}  {'Δ vs first':>11}"]
    for (index, label, points), values in zip(curves, mappings):
        value = values[latest]
        row = f"  {label:<{width}}  {value:9.5f}  {value - baseline:+11.5f}"
        if points[-1][0] > latest:
            row += f"  (later results through {x_axis} {points[-1][0]:g})"
        lines.append(paint(row, index, color))
    lines.append(
        "  Δ is arithmetic, not a better/worse indicator (loss is lower-better)."
    )
    return "\n".join(lines)


def half_block(top, bottom, color):
    """Two independent foreground/background pixels, using terminal default for gaps."""
    if top is not None and bottom is not None:
        if not color:
            return "█"
        return f"\033[38;5;{COLORS[top % len(COLORS)]};48;5;{COLORS[bottom % len(COLORS)]}m▀\033[0m"
    owner = top if top is not None else bottom
    if owner is None:
        return " "
    return paint("▀" if top is not None else "▄", owner, color)


def render_panel(metric, curves, width=100, height=16, color=True, x_axis="epoch"):
    """Plot actual observations at two colored vertical pixels per cell."""
    if not curves:
        return f"\n{metric}: no data\n"
    cols, rows = max(20, width - 12), max(6, height)
    xmax = max(x for _, _, points in curves for x, _ in points)
    xmin = min(x for _, _, points in curves for x, _ in points)
    ymin = min(y for _, _, points in curves for _, y in points)
    ymax = max(y for _, _, points in curves for _, y in points)
    pad = (ymax - ymin) * 0.06 or max(abs(ymax) * 0.01, 0.001)
    ymin, ymax = ymin - pad, ymax + pad
    xmax = max(xmax, xmin + 1)
    # Show half-epoch ticks when present; thin on narrow/wide plots.
    xs = [x for _, _, points in curves for x, _ in points]
    unit = 0.25 if any(x % 0.5 for x in xs) else (0.5 if any(x % 1 for x in xs) else 1)
    tick_step = max(unit, math.ceil((xmax - xmin) / max(1, cols // 5) / unit) * unit)
    ticks = {}
    for tick in range(math.ceil(xmin / tick_step), math.floor(xmax / tick_step) + 1):
        value = tick * tick_step
        col = round((value - xmin) / (xmax - xmin) * (cols - 1))
        ticks[col] = f"{value:g}"
    row_points = {}
    for index, _, points in curves:
        for x, y in points:
            col = round((x - xmin) / (xmax - xmin) * (cols - 1))
            row = round((ymax - y) / (ymax - ymin) * (2 * rows - 1))
            row_points.setdefault(row, []).append((col, index))
    cells = {}
    for row, points in row_points.items():
        if len(points) > cols:
            return f"\n{metric}: too many points on one row; increase --width or zoom with --start/--end\n"
        previous = -1
        for i, (desired, owner) in enumerate(sorted(points)):
            # Preserve left-to-right ordering and reserve space for remaining
            # markers, including collisions against the right-hand boundary.
            col = min(max(desired, previous + 1), cols - (len(points) - i))
            cells[row, col] = owner
            previous = col
    note = "half-block points; collisions offset horizontally, values unchanged"
    lines = [f"\n{metric}  ({note})"]
    for row in range(rows):
        tick = ymax - (2 * row + 0.5) / (2 * rows - 1) * (ymax - ymin)
        axis = f"{tick:9.4f} │" if row in (0, rows // 2, rows - 1) else "          │"
        line = ""
        for col in range(cols):
            top, bottom = cells.get((2 * row, col)), cells.get((2 * row + 1, col))
            occupied = top is not None or bottom is not None
            char = (
                half_block(top, bottom, color)
                if occupied
                else ("┊" if col in ticks else " ")
            )
            if not occupied and col in ticks and color:
                char = f"\033[38;5;240m{char}\033[0m"
            line += char
        lines.append(axis + line)
    lines.append(
        "          └" + "".join("┬" if col in ticks else "─" for col in range(cols))
    )
    middle = "epoch (completed)" if x_axis == "epoch" else "MLflow step"
    labels = [" "] * cols
    for col, label in ticks.items():
        start = max(0, min(cols - len(label), col - len(label) // 2))
        labels[start : start + len(label)] = label
    lines.append("           " + "".join(labels))
    lines.append("           " + middle.center(cols))
    for index, label, points in curves:
        low = min(points, key=lambda p: p[1])
        high = max(points, key=lambda p: p[1])
        lines.append(
            paint(f"  ▀ {label}", index, color)
            + (
                f"  last {points[-1][1]:.4f} @ {points[-1][0]:g}"
                f"  min {low[1]:.4f} @ {low[0]:g}  max {high[1]:.4f} @ {high[0]:g}"
            )
        )
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Exact run names or MLflow run IDs")
    parser.add_argument(
        "-m",
        "--metric",
        "--metrics",
        nargs="+",
        action="extend",
        help="Metric names (repeatable); default: val_metric",
    )
    parser.add_argument("--fold", type=int, help="Restrict to a fold")
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI"),
    )
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--list-metrics", action="store_true")
    parser.add_argument("--x-axis", choices=("epoch", "step"), default="epoch")
    parser.add_argument(
        "--start", type=float, help="First epoch/step to display (zoom)"
    )
    parser.add_argument("--end", type=float, help="Last epoch/step to display (zoom)")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=14, help="Rows per chart")
    color_group = parser.add_mutually_exclusive_group()
    color_group.add_argument("--no-color", action="store_true")
    color_group.add_argument("--color", action="store_true", help="Force ANSI colors")
    parser.add_argument(
        "--watch", type=float, metavar="SECONDS", help="Refresh until Ctrl-C"
    )
    parser.add_argument(
        "--output", type=Path, help="Also save the latest plain-text charts"
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.watch is not None and args.watch < 1:
        parser.error("--watch must be at least 1 second")
    if args.height < 6 or (args.width is not None and args.width < 40):
        parser.error("--height must be >= 6 and --width >= 40")
    if args.start is not None and args.end is not None and args.start > args.end:
        parser.error("--start must be <= --end")
    if not args.tracking_uri:
        parser.error("--tracking-uri or MLFLOW_TRACKING_URI is required")
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=args.tracking_uri)
    metrics = list(dict.fromkeys(args.metric or ["val_metric"]))
    color = args.color or (
        sys.stdout.isatty() and not args.no_color and "NO_COLOR" not in os.environ
    )
    try:
        while True:
            runs = resolve_runs(client, args.runs, args.experiment_name, args.fold)
            if args.list_metrics:
                for run in runs:
                    print(
                        f"{run.info.run_id} {run.data.tags.get('mlflow.runName', '')}"
                    )
                    print("\n".join(f"  {key}" for key in sorted(run.data.metrics)))
                return 0
            panels, warnings = load_curves(
                client, runs, metrics, args.x_axis, args.start, args.end
            )
            width = args.width or shutil.get_terminal_size((100, 30)).columns
            header = "Validation curves · " + time.strftime("%Y-%m-%d %H:%M:%S")
            rendered = (
                header
                + "\n"
                + "\n".join(
                    paint(
                        f"{run.data.tags.get('mlflow.runName', run.info.run_id).rsplit('/', 1)[-1]} "
                        f"f{run.data.params.get('fold', '?')} [{run.info.run_id[:7]}]: "
                        + validation_eta(client, run),
                        index,
                        color,
                    )
                    for index, run in enumerate(runs)
                )
                + "\n"
                + "\n\n".join(
                    render_latest_comparison(metric, curves, color, args.x_axis)
                    for metric, curves in panels.items()
                )
                + "\n"
                + "\n".join(
                    render_panel(metric, curves, width, args.height, color, args.x_axis)
                    for metric, curves in panels.items()
                )
            )
            if warnings:
                rendered += "\n\nWarnings:\n" + "\n".join(warnings)
            if args.watch and sys.stdout.isatty():
                print("\033[2J\033[H", end="")
            print(rendered, flush=True)
            if args.output:
                args.output.write_text(re.sub(r"\x1b\[[0-9;]*m", "", rendered) + "\n")
            if not any(panels.values()) and not (
                args.watch and any(run.info.status == "RUNNING" for run in runs)
            ):
                return 1
            if args.watch is None:
                return 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        # Avoid dumping connection strings / credentials from remote exceptions.
        print(
            f"Unable to render curves ({type(exc).__name__}). Check run IDs, "
            "metrics, tracking connection, and output path.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
