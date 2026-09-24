#!/usr/bin/env python3
"""Plot activity transition rates by location and social-contact context."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "location_zone",
    "social_contact",
    "activity_from",
    "activity_to",
    "n_transitions",
    "n_opportunities",
    "transition_rate_per_frame",
}

TRANSITION_ORDER = ["0_to_1", "1_to_0", "0_to_0", "1_to_1"]
TRANSITION_TITLES = {
    "0_to_1": "Inactive to Active",
    "1_to_0": "Active to Inactive",
    "0_to_0": "Inactive to Inactive",
    "1_to_1": "Active to Active",
}
TRANSITION_SUBTITLES = {
    "0_to_1": "activation",
    "1_to_0": "deactivation",
    "0_to_0": "remains inactive",
    "1_to_1": "remains active",
}
LOCATION_ORDER = [
    "on_nest_threshold",
    "off_nest_threshold",
    "unknown_nest_threshold",
    "nest",
    "foraging",
    "arena",
    "neither",
    "unknown",
]
LOCATION_LABELS = {
    "on_nest_threshold": "On nest",
    "off_nest_threshold": "Off nest",
    "unknown_nest_threshold": "Unknown nest",
    "nest": "Nest",
    "foraging": "Foraging",
    "arena": "Arena",
    "neither": "Neither",
    "unknown": "Unknown",
}
CONTACT_ORDER = ["No social contact", "Social contact", "Unknown contact"]
CONTACT_COLORS = {
    "No social contact": "#4C78A8",
    "Social contact": "#F58518",
    "Unknown contact": "#6B7280",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a grouped bar chart from activity transition summary CSV."
    )
    parser.add_argument(
        "csv",
        type=Path,
        help="Path to activity_transition_rates_by_location_contact.csv.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help=(
            "Output path without extension. Defaults to the CSV path with "
            "'_barchart' appended."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--no-separate",
        action="store_true",
        help="Only write the combined 2x2 figure; by default separate per-transition figures are also written.",
    )
    return parser.parse_args()


def contact_label(value: object) -> str:
    if pd.isna(value):
        return "Unknown contact"
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return "Social contact"
    if text in {"false", "0", "no"}:
        return "No social contact"
    return "Unknown contact"


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    data = data.copy()
    if "transition_label" not in data:
        data["transition_label"] = (
            data["activity_from"].astype(int).astype(str)
            + "_to_"
            + data["activity_to"].astype(int).astype(str)
        )
    data["rate_percent"] = data["transition_rate_per_frame"] * 100.0
    data["location_zone"] = data["location_zone"].fillna("unknown").astype(str)
    data["location_label"] = data["location_zone"].map(LOCATION_LABELS).fillna(
        data["location_zone"].str.replace("_", " ").str.title()
    )
    data["contact_label"] = data["social_contact"].map(contact_label)
    return data


def ordered_values(values: pd.Series, preferred: list[str]) -> list[str]:
    present = list(dict.fromkeys(values.dropna().astype(str)))
    ordered = [value for value in preferred if value in present]
    ordered.extend(sorted(value for value in present if value not in ordered))
    return ordered


def count_pair_label(events: int, opportunities: int) -> str:
    return f"{events:,}/{opportunities:,}"


def transition_axis_limits(values: pd.Series) -> tuple[float, float]:
    max_value = float(values.max())
    min_value = float(values.min())
    if min_value > 80:
        span = max(max_value - min_value, 0.5)
        y_min = max(0.0, np.floor((min_value - span * 0.6) * 10.0) / 10.0)
        y_max = min(100.0, np.ceil((max_value + span * 0.8) * 10.0) / 10.0)
        if y_max - y_min < 3.0:
            center = (y_min + y_max) / 2.0
            y_min = max(0.0, center - 1.5)
            y_max = min(100.0, center + 1.5)
        return y_min, y_max
    if max_value <= 8:
        return 0.0, max(2.0, np.ceil(max_value * 1.35 * 10.0) / 10.0)
    return 0.0, min(100.0, np.ceil(max_value * 1.25))


def draw_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    transition_label: str,
    locations: list[str],
    contacts: list[str],
) -> None:
    panel = data[data["transition_label"] == transition_label]
    x_base = np.arange(len(locations))
    group_width = 0.76
    bar_width = group_width / max(len(contacts), 1)
    y_min, y_max = transition_axis_limits(panel["rate_percent"])

    for index, contact in enumerate(contacts):
        x_values = x_base - group_width / 2 + (index + 0.5) * bar_width
        y_values = []
        counts = []
        for location in locations:
            row = panel[
                (panel["location_zone"] == location)
                & (panel["contact_label"] == contact)
            ]
            if row.empty:
                y_values.append(np.nan)
                counts.append(None)
            else:
                y_values.append(float(row.iloc[0]["rate_percent"]))
                counts.append(
                    (
                        int(row.iloc[0]["n_transitions"]),
                        int(row.iloc[0]["n_opportunities"]),
                    )
                )

        bars = ax.bar(
            x_values,
            y_values,
            width=bar_width * 0.88,
            color=CONTACT_COLORS.get(contact, "#6B7280"),
            edgecolor="#222222",
            linewidth=0.5,
            label=contact,
        )
        for bar, value, count in zip(bars, y_values, counts, strict=False):
            if np.isnan(value):
                continue
            label_y = value + (y_max - y_min) * 0.025
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.2,
                color="#222222",
            )

    title = TRANSITION_TITLES.get(transition_label, transition_label)
    subtitle = TRANSITION_SUBTITLES.get(transition_label, "")
    ax.set_title(f"{title}\n{subtitle}", fontsize=12)
    ax.set_xticks(x_base)
    ax.set_xticklabels([LOCATION_LABELS.get(item, item.title()) for item in locations])
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Next-frame outcome rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis="y", color="#D7DCE2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if y_min > 0:
        ax.text(
            0.02,
            0.03,
            "zoomed y-axis",
            transform=ax.transAxes,
            fontsize=7.5,
            color="#555555",
            ha="left",
            va="bottom",
        )


def draw_separate_transition(
    data: pd.DataFrame,
    transition_label: str,
    locations: list[str],
    contacts: list[str],
    output_prefix: Path,
    dpi: int,
) -> None:
    panel = data[data["transition_label"] == transition_label]
    x_base = np.arange(len(locations))
    group_width = 0.72
    bar_width = group_width / max(len(contacts), 1)
    y_min, y_max = transition_axis_limits(panel["rate_percent"])
    label_pad = (y_max - y_min) * 0.03

    fig, ax = plt.subplots(figsize=(12.5, 7.2), constrained_layout=False)
    count_rows: list[list[str]] = []
    for index, contact in enumerate(contacts):
        x_values = x_base - group_width / 2 + (index + 0.5) * bar_width
        y_values = []
        count_labels = []
        for location in locations:
            row = panel[
                (panel["location_zone"] == location)
                & (panel["contact_label"] == contact)
            ]
            if row.empty:
                y_values.append(np.nan)
                count_labels.append("")
            else:
                y_values.append(float(row.iloc[0]["rate_percent"]))
                count_labels.append(
                    count_pair_label(
                        int(row.iloc[0]["n_transitions"]),
                        int(row.iloc[0]["n_opportunities"]),
                    )
                )

        bars = ax.bar(
            x_values,
            y_values,
            width=bar_width * 0.88,
            color=CONTACT_COLORS.get(contact, "#6B7280"),
            edgecolor="#222222",
            linewidth=0.6,
            label=contact,
        )
        count_rows.append(count_labels)
        for bar, value in zip(bars, y_values, strict=False):
            if np.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(y_max - label_pad * 0.2, value + label_pad),
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#222222",
            )

    title = TRANSITION_TITLES.get(transition_label, transition_label)
    subtitle = TRANSITION_SUBTITLES.get(transition_label, "")
    ax.set_title(f"{title}: {subtitle}", fontsize=15, fontweight="bold", pad=16)
    ax.set_xticks(x_base)
    ax.set_xticklabels([LOCATION_LABELS.get(item, item.title()) for item in locations])
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Next-frame outcome rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1))
    ax.grid(axis="y", color="#D7DCE2", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if y_min > 0:
        ax.text(
            0.01,
            0.97,
            "Zoomed y-axis to show differences near 100%",
            transform=ax.transAxes,
            fontsize=9,
            color="#555555",
            ha="left",
            va="top",
        )

    table = ax.table(
        cellText=count_rows,
        rowLabels=contacts,
        colLabels=[LOCATION_LABELS.get(item, item.title()) for item in locations],
        cellLoc="center",
        rowLoc="right",
        bbox=[0.0, -0.48, 1.0, 0.30],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for cell in table.get_celld().values():
        cell.set_edgecolor("#D7DCE2")
        cell.set_linewidth(0.6)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(contacts),
        frameon=False,
        title="Social-contact status on the starting frame",
    )
    fig.text(
        0.5,
        0.055,
        "Table entries are raw events/opportunities for each bar.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(top=0.84, bottom=0.38, left=0.08, right=0.985)

    fig.savefig(output_prefix.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot(
    data: pd.DataFrame,
    output_prefix: Path,
    dpi: int,
    write_separate: bool = True,
) -> list[Path]:
    transitions = [item for item in TRANSITION_ORDER if item in set(data["transition_label"])]
    if not transitions:
        raise ValueError("No recognized transition labels found in the CSV.")

    locations = ordered_values(data["location_zone"], LOCATION_ORDER)
    contacts = ordered_values(data["contact_label"], CONTACT_ORDER)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "figure.titlesize": 16,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=False)
    axes_flat = list(axes.ravel())

    for ax, transition in zip(axes_flat, transitions, strict=False):
        draw_panel(ax, data, transition, locations, contacts)
    for ax in axes_flat[len(transitions) :]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=len(labels),
        frameon=False,
        title="Social-contact status on the starting frame",
    )
    fig.suptitle(
        "Activity Transitions by Location and Social Contact",
        y=0.985,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.045,
        (
            "Each bar corresponds to one summarized CSV row: location_zone + social_contact + "
            "activity_from -> activity_to. Separate figures include raw event/opportunity counts."
        ),
        ha="center",
        fontsize=10,
        color="#333333",
    )
    fig.text(
        0.5,
        0.018,
        (
            "Activity coding: 0 = inactive, 1 = active. Percent labels show next-frame outcome rate. "
            "Persistence panels use zoomed y-axes."
        ),
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(top=0.84, bottom=0.13, left=0.07, right=0.985, hspace=0.42, wspace=0.18)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=dpi)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)

    separate_prefixes = []
    if write_separate:
        for transition in transitions:
            separate_prefix = output_prefix.with_name(f"{output_prefix.name}_{transition}")
            draw_separate_transition(
                data,
                transition,
                locations,
                contacts,
                separate_prefix,
                dpi,
            )
            separate_prefixes.append(separate_prefix)
    return separate_prefixes


def main() -> int:
    args = parse_args()
    csv_path = args.csv.expanduser().resolve()
    if args.output_prefix is None:
        output_prefix = csv_path.with_name(f"{csv_path.stem}_barchart")
    else:
        output_prefix = args.output_prefix.expanduser().resolve()

    data = load_data(csv_path)
    separate_prefixes = plot(data, output_prefix, args.dpi, write_separate=not args.no_separate)
    print(f"Wrote {output_prefix.with_suffix('.png')}")
    print(f"Wrote {output_prefix.with_suffix('.pdf')}")
    if not args.no_separate:
        for separate_prefix in separate_prefixes:
            print(f"Wrote {separate_prefix.with_suffix('.png')}")
            print(f"Wrote {separate_prefix.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
