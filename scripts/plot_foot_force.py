#!/usr/bin/env python3

"""Plot the three foot-force measurements recorded during play."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


METHODS = (
    ("magnitude", "Contact-force magnitude"),
    ("vertical", "Positive world-Z contact force"),
    ("joint_wrench_x_div100", "Joint-wrench body-X / 100"),
)
COMMANDS = (
    ("cmd_lin_vel_x", "Linear X"),
    ("cmd_lin_vel_y", "Linear Y"),
    ("cmd_ang_vel_z", "Angular Z"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="CSV created by scripts/rsl_rl/play.py")
    parser.add_argument("--output", type=Path, help="Output image path; defaults beside the CSV")
    parser.add_argument("--show", action="store_true", help="Also open the interactive plot window")
    return parser.parse_args()


def read_recording(csv_path: Path) -> tuple[list[float], dict[str, list[float]]]:
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"Recording has no header: {csv_path}")

        expected = [f"{method}_foot_{foot}" for method, _ in METHODS for foot in range(4)]
        expected.extend(column for column, _ in COMMANDS)
        missing = [column for column in ["time_s", *expected] if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Recording is missing columns: {', '.join(missing)}")

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
    times, values = read_recording(args.csv_path)
    output_path = args.output or args.csv_path.with_suffix(".png")

    figure, axes = plt.subplots(4, 1, figsize=(25, 15), sharex=True)
    for axis, (method, title) in zip(axes[:3], METHODS):
        for foot in range(4):
            axis.plot(times, values[f"{method}_foot_{foot}"], label=f"Foot {foot}", linewidth=1.0)
        axis.set_title(title)
        axis.set_ylabel("Force value")
        axis.grid(alpha=0.3)
        axis.legend(ncol=4)

    for column, label in COMMANDS:
        axes[3].plot(times, values[column], label=label, linewidth=1.0)
    axes[3].set_title("Commanded base velocity")
    axes[3].set_ylabel("Command")
    axes[3].grid(alpha=0.3)
    axes[3].legend(ncol=3)

    axes[-1].set_xlabel("Simulation time (s)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    print(f"Saved plot to: {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
