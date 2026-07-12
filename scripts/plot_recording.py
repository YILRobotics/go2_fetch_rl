#!/usr/bin/env python3

"""Plot foot force and high-level observations recorded during play."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


FOOT_FORCE_METHODS = (
    ("normal_force", "Foot normal force"),
    ("magnitude", "Contact-force magnitude"),
    ("vertical", "Positive world-Z contact force"),
    ("joint_wrench_x_div100", "Joint-wrench body-X / 100"),
)
COMMANDS = (
    ("cmd_lin_vel_x", "Linear X"),
    ("cmd_lin_vel_y", "Linear Y"),
    ("cmd_ang_vel_z", "Angular Z"),
)
OBSERVATIONS = (
    (("cube_pos_obs_x", "X"), ("cube_pos_obs_y", "Y"), "Cube position observation", "Position (m)"),
    (("cube_vel_obs_x", "X"), ("cube_vel_obs_y", "Y"), "Cube velocity observation", "Velocity (m/s)"),
)
HIGH_LEVEL_DT_S = 0.065
DEFAULT_LOGS_DIR = Path(__file__).resolve().parents[1] / "logs"


def high_level_sample_indices(times: list[float]) -> list[int]:
    """Select one CSV row per 15.38 Hz high-level control interval."""
    indices: list[int] = []
    previous_interval: int | None = None
    for index, time_s in enumerate(times):
        interval = int((time_s + 1.0e-9) / HIGH_LEVEL_DT_S)
        if interval != previous_interval:
            indices.append(index)
            previous_interval = interval
    return indices


def grouped_observation_columns(values: dict[str, list[float]]) -> list[tuple[str, list[tuple[str, str]]]]:
    """Group ``obs__GROUP__TERM__INDEX`` columns into plotting panels."""
    grouped: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for column in values:
        parts = column.split("__")
        if len(parts) != 4 or parts[0] != "obs":
            continue
        group, term, component_text = parts[1:]
        try:
            component = int(component_text)
        except ValueError:
            continue
        grouped[(group, term)].append((component, column))

    panels = []
    policy_terms = {term for group, term in grouped if group == "policy"}
    for (group, term), columns in grouped.items():
        # Shared critic terms duplicate the corresponding policy plots.
        if group == "critic" and term in policy_terms:
            continue
        labeled_columns = [(column, f"Component {component}") for component, column in sorted(columns)]
        panels.append((f"{group}: {term}", labeled_columns))
    return panels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        help="CSV created by scripts/rsl_rl/play.py; defaults to the newest CSV under logs/",
    )
    parser.add_argument("--output", type=Path, help="Output image path; defaults beside the CSV")
    parser.add_argument("--show", action="store_true", help="Also open the interactive plot window")
    return parser.parse_args()


def newest_recording(logs_dir: Path = DEFAULT_LOGS_DIR) -> Path:
    """Return the most recently modified CSV anywhere below the logs directory."""
    recordings = list(logs_dir.rglob("*.csv"))
    if not recordings:
        raise FileNotFoundError(f"No CSV recordings found under: {logs_dir}")
    return max(recordings, key=lambda path: path.stat().st_mtime)


def read_recording(csv_path: Path) -> tuple[list[float], dict[str, list[float]]]:
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"Recording has no header: {csv_path}")

        if "time_s" not in reader.fieldnames:
            raise ValueError("Recording is missing column: time_s")
        expected = [column for column in reader.fieldnames if column not in ("step", "time_s")]

        times: list[float] = []
        values = {column: [] for column in expected}
        for row in reader:
            times.append(float(row["time_s"]))
            for column in expected:
                values[column].append(float(row[column]))

    if not times:
        raise ValueError(f"Recording contains no samples: {csv_path}")
    return times, values


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path or newest_recording()
    if args.csv_path is None:
        print(f"Using newest CSV: {csv_path}")
    times, values = read_recording(csv_path)
    output_path = args.output or csv_path.with_suffix(".png")

    force_methods = [
        (method, title)
        for method, title in FOOT_FORCE_METHODS
        if all(f"{method}_foot_{foot}" in values for foot in range(4))
    ]
    # Low-level measurements are plotted first at their native 50 Hz rate.
    panels = [("force", item) for item in force_methods]
    # Commands and manager observations update only at the high-level policy rate.
    if all(column in values for column, _ in COMMANDS):
        panels.append(("command", COMMANDS))
    panels.extend(
        ("observation", observation)
        for observation in OBSERVATIONS
        if all(column in values for column, _ in observation[:2])
    )
    panels.extend(
        ("manager_observation", observation_panel)
        for observation_panel in grouped_observation_columns(values)
    )
    if not panels:
        raise ValueError("Recording contains no supported plot columns")

    high_level_indices = high_level_sample_indices(times)
    high_level_times = [times[index] for index in high_level_indices]
    figure, axes = plt.subplots(len(panels), 1, figsize=(35, 3.7 * len(panels)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for axis, (panel_type, panel) in zip(axes, panels):
        if panel_type == "force":
            method, title = panel
            for foot in range(4):
                axis.plot(times, values[f"{method}_foot_{foot}"], label=f"Foot {foot}", linewidth=1.0)
            axis.set_title(f"50 Hz — {title}")
            axis.set_ylabel("Force (N)")
            axis.legend(ncol=4)
        elif panel_type == "command":
            for column, label in COMMANDS:
                samples = [values[column][index] for index in high_level_indices]
                axis.plot(high_level_times, samples, label=label, linewidth=1.0)
            axis.set_title("15.38 Hz — Commanded base velocity")
            axis.set_ylabel("Command")
            axis.legend(ncol=3)
        elif panel_type == "observation":
            x_item, y_item, title, ylabel = panel
            for column, label in (x_item, y_item):
                samples = [values[column][index] for index in high_level_indices]
                axis.plot(high_level_times, samples, label=label, linewidth=1.0)
            axis.set_title(f"15.38 Hz — {title}")
            axis.set_ylabel(ylabel)
            axis.legend(ncol=2)
        else:
            title, columns = panel
            for column, label in columns:
                samples = [values[column][index] for index in high_level_indices]
                axis.plot(high_level_times, samples, label=label, linewidth=1.0)
            axis.set_title(f"15.38 Hz — {title}")
            axis.set_ylabel("Observation")
            axis.legend(ncol=min(4, len(columns)))
        axis.grid(alpha=0.3)

    axes[-1].set_xlabel("Simulation time (s)")
    figure.suptitle(
        "Play recording: 50 Hz low-level measurements, then 15.38 Hz high-level observations",
        fontsize=24,
        y=0.998,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(output_path, dpi=160)
    print(f"Saved plot to: {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
