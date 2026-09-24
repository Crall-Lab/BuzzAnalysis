#!/usr/bin/env python3
"""Plot inactive-to-active transition rate over time and colony size."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
CONTACT_COLORS = {
    "No social contact": "#4C78A8",
    "Social contact": "#F58518",
    "Unknown contact": "#6B7280",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create inactive-to-active rate plots from the excluded-tag summary CSV."
    )
    parser.add_argument(
        "csv",
        type=Path,
        help="Path to inactive_to_active_rates_by_date_location_contact.csv.",
    )
    parser.add_argument("--fps", type=float, default=4.5)
    parser.add_argument("--min-opportunities", type=int, default=50)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="Output path without extension. Defaults to the CSV path with '_over_time' appended.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--no-separate",
        action="store_true",
        help="Only write the combined figure; by default separate per-location figures are also written.",
    )
    parser.add_argument(
        "--label-counts",
        action="store_true",
        help=(
            "Annotate the time-series points in separate per-location figures with "
            "raw inactive-to-active events/inactive-frame opportunities."
        ),
    )
    parser.add_argument(
        "--count-bars",
        action="store_true",
        help=(
            "Write one all-points bar chart per location/contact status, with every "
            "bar labeled as raw inactive-to-active events/inactive-frame opportunities."
        ),
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


def load_and_collapse(path: Path, fps: float, min_opportunities: int) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {
        "Date",
        "location_zone",
        "social_contact",
        "n_inactive_opportunities",
        "n_inactive_to_active",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    data = data.copy()
    if "true_colony_size" not in data:
        data["true_colony_size"] = np.nan
    data["Date"] = pd.to_datetime(data["Date"])
    data["location_zone"] = data["location_zone"].fillna("unknown").astype(str)
    data["contact_label"] = data["social_contact"].map(contact_label)

    group_cols = ["Date", "location_zone", "contact_label", "true_colony_size"]
    collapsed = (
        data.groupby(group_cols, dropna=False)
        .agg(
            n_inactive_opportunities=("n_inactive_opportunities", "sum"),
            n_inactive_to_active=("n_inactive_to_active", "sum"),
        )
        .reset_index()
    )
    collapsed = collapsed.loc[collapsed["n_inactive_opportunities"] >= min_opportunities].copy()
    collapsed["inactive_to_active_rate_per_frame"] = (
        collapsed["n_inactive_to_active"] / collapsed["n_inactive_opportunities"]
    )
    collapsed["inactive_to_active_rate_per_second"] = (
        collapsed["inactive_to_active_rate_per_frame"] * float(fps)
    )
    collapsed["rate_percent_per_frame"] = collapsed["inactive_to_active_rate_per_frame"] * 100.0
    return collapsed


def ordered_locations(values: pd.Series) -> list[str]:
    present = list(dict.fromkeys(values.dropna().astype(str)))
    ordered = [value for value in LOCATION_ORDER if value in present]
    ordered.extend(sorted(value for value in present if value not in ordered))
    return ordered


def rate_axis_limits(values: pd.Series) -> tuple[float, float]:
    values = values.dropna()
    if values.empty:
        return 0.0, 1.0
    min_value = float(values.min())
    max_value = float(values.max())
    span = max(max_value - min_value, 0.5)
    y_min = max(0.0, min_value - span * 0.15)
    y_max = max_value + span * 0.20
    if y_max - y_min < 2.0:
        center = (y_min + y_max) / 2.0
        y_min = max(0.0, center - 1.0)
        y_max = center + 1.0
    return y_min, y_max


def count_label(row: pd.Series) -> str:
    events = int(row["n_inactive_to_active"])
    opportunities = int(row["n_inactive_opportunities"])
    return f"{events:,}/{opportunities:,}"


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def annotate_time_counts(
    ax: plt.Axes,
    contact_data: pd.DataFrame,
    contact: str,
    color: str,
) -> None:
    offsets = {
        "No social contact": (0, 8, "bottom"),
        "Social contact": (0, -11, "top"),
        "Unknown contact": (0, 16, "bottom"),
    }
    dx, dy, va = offsets.get(contact, (0, 8, "bottom"))
    for index, row in contact_data.reset_index(drop=True).iterrows():
        # Alternate a small horizontal nudge so same-day labels do not sit exactly on top of each other.
        x_nudge = dx + (-5 if index % 2 else 5)
        ax.annotate(
            count_label(row),
            xy=(row["Date"], row["rate_percent_per_frame"]),
            xytext=(x_nudge, dy),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=5.4,
            color="#222222",
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.35,
                "alpha": 0.86,
            },
            clip_on=False,
        )


def draw_location_panel(
    ax_time: plt.Axes,
    ax_size: plt.Axes,
    panel: pd.DataFrame,
    contacts: list[str],
    y_limits: tuple[float, float],
    label_counts: bool = False,
) -> None:
    for contact in contacts:
        contact_data = panel.loc[panel["contact_label"] == contact].sort_values("Date")
        if contact_data.empty:
            continue
        color = CONTACT_COLORS.get(contact, "#6B7280")
        ax_time.plot(
            contact_data["Date"],
            contact_data["rate_percent_per_frame"],
            marker="o",
            markersize=3.2,
            linewidth=1.4,
            color=color,
            label=contact,
        )
        if label_counts:
            annotate_time_counts(ax_time, contact_data, contact, color)
        with_size = contact_data.loc[contact_data["true_colony_size"].notna()]
        ax_size.scatter(
            with_size["true_colony_size"],
            with_size["rate_percent_per_frame"],
            s=np.clip(with_size["n_inactive_opportunities"] / 250.0, 14, 90),
            alpha=0.82,
            color=color,
            edgecolor="#222222",
            linewidth=0.3,
            label=contact,
        )

    for ax in (ax_time, ax_size):
        ax.set_ylim(*y_limits)
        ax.grid(axis="y", color="#D7DCE2", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_time.tick_params(axis="x", rotation=35)
    ax_size.set_xlabel("Biological colony size")


def plot_location_figure(
    panel: pd.DataFrame,
    location: str,
    contacts: list[str],
    output_prefix: Path,
    dpi: int,
    label_counts: bool = False,
) -> None:
    title = LOCATION_LABELS.get(location, location.replace("_", " ").title())
    y_limits = rate_axis_limits(panel["rate_percent_per_frame"])
    n_dates = max(1, panel["Date"].nunique())
    figure_width = min(24.0, max(11.5, n_dates * 0.34))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(figure_width, 8.2 if label_counts else 7.4),
        sharey=True,
        constrained_layout=False,
    )
    draw_location_panel(axes[0], axes[1], panel, contacts, y_limits, label_counts=label_counts)
    axes[0].set_title(f"{title}: over time", fontsize=13, fontweight="bold")
    axes[1].set_title(f"{title}: by biological colony size", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Inactive to active rate\n(% of inactive-frame opportunities)")
    axes[1].set_ylabel("Inactive to active rate\n(% of inactive-frame opportunities)")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncols=len(handles),
            frameon=False,
            title="Social-contact status on the starting frame",
        )
    fig.suptitle("Inactive-to-Active Transition Rate", y=1.035, fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        0.012,
        (
            "Time-panel labels are raw inactive-to-active events/inactive-frame opportunities. "
            "Point size in the colony-size panel reflects the number of inactive-frame opportunities."
            if label_counts
            else "Point size in the colony-size panel reflects the number of inactive-frame opportunities."
        ),
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(top=0.83, bottom=0.10, left=0.10, right=0.985, hspace=0.38)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot(
    data: pd.DataFrame,
    output_prefix: Path,
    dpi: int,
    write_separate: bool = True,
    label_counts: bool = False,
) -> list[Path]:
    locations = ordered_locations(data["location_zone"])
    contacts = [
        value
        for value in ["No social contact", "Social contact", "Unknown contact"]
        if value in set(data["contact_label"])
    ]

    ncols = max(1, len(locations))
    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(4.4 * ncols, 7.2),
        sharey=False,
        squeeze=False,
        constrained_layout=True,
    )

    for col, location in enumerate(locations):
        panel = data.loc[data["location_zone"] == location].copy()
        ax_time = axes[0, col]
        ax_size = axes[1, col]
        y_limits = rate_axis_limits(panel["rate_percent_per_frame"])
        draw_location_panel(ax_time, ax_size, panel, contacts, y_limits)

        title = LOCATION_LABELS.get(location, location.replace("_", " ").title())
        ax_time.set_title(title)

    axes[0, 0].set_ylabel("Inactive to active rate\n(% of inactive-frame opportunities)")
    axes[1, 0].set_ylabel("Inactive to active rate\n(% of inactive-frame opportunities)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.035),
            ncols=len(handles),
            frameon=False,
            title="Social-contact status on the starting frame",
        )
    fig.suptitle("Inactive-to-Active Transition Rate by Location and Social Contact", y=1.09)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    separate_prefixes: list[Path] = []
    if write_separate:
        for location in locations:
            panel = data.loc[data["location_zone"] == location].copy()
            if panel.empty:
                continue
            separate_prefix = output_prefix.with_name(f"{output_prefix.name}_{location}")
            plot_location_figure(
                panel,
                location,
                contacts,
                separate_prefix,
                dpi,
                label_counts=label_counts,
            )
            separate_prefixes.append(separate_prefix)
    return separate_prefixes


def plot_count_bar_figures(data: pd.DataFrame, output_prefix: Path, dpi: int) -> list[Path]:
    written: list[Path] = []
    locations = ordered_locations(data["location_zone"])
    contacts = [
        value
        for value in ["No social contact", "Social contact", "Unknown contact"]
        if value in set(data["contact_label"])
    ]

    for location in locations:
        for contact in contacts:
            panel = data.loc[
                (data["location_zone"] == location) & (data["contact_label"] == contact)
            ].sort_values("Date")
            if panel.empty:
                continue

            n_bars = len(panel)
            figure_width = min(26.0, max(11.0, n_bars * 0.34))
            y_max = max(1.0, float(panel["rate_percent_per_frame"].max()))
            y_limit = y_max * 1.28 if y_max >= 5.0 else y_max + 2.5
            x = np.arange(n_bars)
            color = CONTACT_COLORS.get(contact, "#6B7280")
            title = LOCATION_LABELS.get(location, location.replace("_", " ").title())

            fig, ax = plt.subplots(figsize=(figure_width, 6.2), constrained_layout=False)
            bars = ax.bar(
                x,
                panel["rate_percent_per_frame"],
                color=color,
                alpha=0.82,
                edgecolor="#222222",
                linewidth=0.35,
            )
            label_pad = y_limit * 0.015
            for bar, (_, row) in zip(bars, panel.iterrows()):
                height = float(bar.get_height())
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + label_pad,
                    count_label(row),
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=6,
                    color="#222222",
                )

            date_labels = panel["Date"].dt.strftime("%Y-%m-%d").tolist()
            tick_step = max(1, int(np.ceil(n_bars / 18)))
            tick_positions = x[::tick_step]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([date_labels[i] for i in tick_positions], rotation=45, ha="right")
            ax.set_ylim(0.0, y_limit)
            ax.set_ylabel("Inactive to active rate\n(% of inactive-frame opportunities)")
            ax.set_xlabel("Date")
            ax.set_title(f"{title} / {contact}: all points", fontsize=13, fontweight="bold")
            ax.grid(axis="y", color="#D7DCE2", linewidth=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.text(
                0.5,
                0.02,
                "Bar labels are raw inactive-to-active events/inactive-frame opportunities.",
                ha="center",
                fontsize=9,
                color="#555555",
            )
            fig.subplots_adjust(top=0.88, bottom=0.24, left=0.08, right=0.985)

            prefix = output_prefix.with_name(
                f"{output_prefix.name}_{slugify(location)}_{slugify(contact)}_count_bars"
            )
            fig.savefig(prefix.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
            fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(fig)
            written.append(prefix)

    return written


def main() -> int:
    args = parse_args()
    csv_path = args.csv.expanduser().resolve()
    output_prefix = (
        args.output_prefix.expanduser().resolve()
        if args.output_prefix
        else csv_path.with_name(csv_path.stem + "_over_time")
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    data = load_and_collapse(csv_path, args.fps, args.min_opportunities)
    data.to_csv(output_prefix.with_name(output_prefix.name + "_collapsed.csv"), index=False)
    separate_prefixes = plot(
        data,
        output_prefix,
        args.dpi,
        write_separate=not args.no_separate,
        label_counts=args.label_counts,
    )
    count_bar_prefixes = (
        plot_count_bar_figures(data, output_prefix, args.dpi) if args.count_bars else []
    )
    print(f"Wrote {output_prefix.with_suffix('.png')}")
    print(f"Wrote {output_prefix.with_suffix('.pdf')}")
    print(f"Wrote {output_prefix.with_name(output_prefix.name + '_collapsed.csv')}")
    for separate_prefix in separate_prefixes:
        print(f"Wrote {separate_prefix.with_suffix('.png')}")
        print(f"Wrote {separate_prefix.with_suffix('.pdf')}")
    for count_bar_prefix in count_bar_prefixes:
        print(f"Wrote {count_bar_prefix.with_suffix('.png')}")
        print(f"Wrote {count_bar_prefix.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
