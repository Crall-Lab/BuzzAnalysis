#!/usr/bin/env python3
"""Summarize activity transitions by on/off nest-threshold status.

This builds a nest-specific analysis from the preliminary frame-level outputs.
It does not modify the frame-level files. Excluded bee IDs are omitted only from
the downstream summaries, and contact context is recomputed after removing
contact pairs involving excluded IDs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from summarize_inactive_to_active import (
    VIDEO_COLS,
    default_colony_size_file,
    excluded_tag_detection_audit,
    fps_from_metadata,
    load_colony_sizes,
    load_contact_index,
    observed_sizes,
    recompute_contacts,
    resolve_index_path,
)


NEST_STATUS_COL = "nest_threshold_status"
PERIMETER_STATUS_COL = "nest_perimeter_status"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build nest-specific activity transition summaries from preliminary BuzzAnalysis outputs."
    )
    parser.add_argument("results_root", type=Path, help="Preliminary results folder.")
    parser.add_argument(
        "--exclude-bee-ids",
        nargs="*",
        type=int,
        default=[],
        help="Bee IDs to exclude from summaries and contact-context recomputation.",
    )
    parser.add_argument("--fps", type=float, help="Override fps; otherwise read from run_metadata.json.")
    parser.add_argument(
        "--colony-size-file",
        type=Path,
        help=(
            "Optional CSV with one row per date and biological colony size. "
            "Defaults to the matching colony-size CSV using run_metadata.json when present."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Folder for outputs; default is results_root/nest_transition_rates_excluding_<ids>.",
    )
    parser.add_argument(
        "--start-date",
        help="Optional first date to include, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        help="Optional last date to include, in YYYY-MM-DD format.",
    )
    return parser.parse_args()


def output_dir_for(results_root: Path, excluded_ids: Iterable[int], requested: Path | None) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    suffix = "_".join(str(bee_id) for bee_id in excluded_ids) or "none"
    return (results_root / f"nest_transition_rates_excluding_{suffix}").resolve()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def filter_frame_index_by_date(
    frame_index: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if frame_index.empty or (start_date is None and end_date is None):
        return frame_index
    work = frame_index.copy()
    dates = pd.to_datetime(work["Date"])
    if start_date is not None:
        dates_start = pd.to_datetime(start_date)
        work = work.loc[dates >= dates_start].copy()
        dates = pd.to_datetime(work["Date"])
    if end_date is not None:
        dates_end = pd.to_datetime(end_date)
        work = work.loc[dates <= dates_end].copy()
    return work.reset_index(drop=True)


def bool_status(series: pd.Series, true_label: str, false_label: str, unknown_label: str) -> pd.Series:
    status = pd.Series(unknown_label, index=series.index, dtype="object")
    valid = series.notna()
    if not valid.any():
        return status
    values = series.loc[valid].astype("boolean")
    status.loc[valid & values.fillna(False)] = true_label
    status.loc[valid & ~values.fillna(False)] = false_label
    return status


def add_nest_status_columns(frame_level: pd.DataFrame) -> pd.DataFrame:
    work = frame_level.copy()
    if "within_nest_perimeter_threshold" in work:
        work[NEST_STATUS_COL] = bool_status(
            work["within_nest_perimeter_threshold"],
            "on_nest_threshold",
            "off_nest_threshold",
            "unknown_nest_threshold",
        )
    else:
        work[NEST_STATUS_COL] = "unknown_nest_threshold"

    if "inside_nest_perimeter" in work:
        work[PERIMETER_STATUS_COL] = bool_status(
            work["inside_nest_perimeter"],
            "inside_nest_perimeter",
            "outside_nest_perimeter",
            "unknown_nest_perimeter",
        )
    else:
        work[PERIMETER_STATUS_COL] = "unknown_nest_perimeter"
    return work


def normalize_social_contact(frame_level: pd.DataFrame) -> pd.Series:
    if "social_contact" not in frame_level:
        return pd.Series("unknown", index=frame_level.index, dtype="object")
    values = frame_level["social_contact"]
    out = pd.Series("unknown", index=frame_level.index, dtype="object")
    valid = values.notna()
    if valid.any():
        bool_values = values.loc[valid].astype("boolean")
        out.loc[valid & bool_values.fillna(False)] = "True"
        out.loc[valid & ~bool_values.fillna(False)] = "False"
    return out


def transition_events(frame_level: pd.DataFrame) -> pd.DataFrame:
    if frame_level.empty:
        return pd.DataFrame()

    work = add_nest_status_columns(frame_level)
    work = work.sort_values(["source_file", "bee_ID", "frame"]).copy()
    grouped = work.groupby(["source_file", "bee_ID"], sort=False)
    work["next_activity"] = grouped["activity"].shift(-1)
    work["next_frame"] = grouped["frame"].shift(-1)
    valid = (
        work["activity"].isin([0, 1])
        & work["next_activity"].isin([0, 1])
        & ((work["next_frame"] - work["frame"]) == 1)
    )
    work = work.loc[valid].copy()
    if work.empty:
        return pd.DataFrame()

    work["activity_from"] = work["activity"].astype(int)
    work["activity_to"] = work["next_activity"].astype(int)
    work["social_contact"] = normalize_social_contact(work)
    return work


def summarize_transition_counts(events: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=[*group_cols, "activity_from", "activity_to", "n_transitions"])
    return (
        events.groupby([*group_cols, "activity_from", "activity_to"], dropna=False)
        .size()
        .reset_index(name="n_transitions")
    )


def add_transition_rates(transitions: pd.DataFrame, fps: float) -> pd.DataFrame:
    if transitions.empty:
        return transitions
    denom_cols = [col for col in transitions.columns if col not in {"activity_to", "n_transitions"}]
    out = transitions.copy()
    out["n_opportunities"] = out.groupby(denom_cols, dropna=False)["n_transitions"].transform("sum")
    out["transition_rate_per_frame"] = out["n_transitions"] / out["n_opportunities"]
    out["transition_rate_per_second"] = out["transition_rate_per_frame"] * float(fps)
    out["transition_label"] = out["activity_from"].astype(str) + "_to_" + out["activity_to"].astype(str)
    return out


def collapse_transition_rates(transitions: pd.DataFrame, group_cols: list[str], fps: float) -> pd.DataFrame:
    if transitions.empty:
        return pd.DataFrame(columns=[*group_cols, "activity_from", "activity_to", "n_transitions",
                                     "n_opportunities", "transition_rate_per_frame", "transition_rate_per_second", "transition_label"])
    counts = (
        transitions.groupby([*group_cols, "activity_from", "activity_to"], dropna=False)["n_transitions"]
        .sum()
        .reset_index()
    )
    return add_transition_rates(counts, fps)


def summarize_inactive_to_active(events: pd.DataFrame, group_cols: list[str], fps: float) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=group_cols)
    inactive = events.loc[events["activity_from"] == 0].copy()
    if inactive.empty:
        return pd.DataFrame(columns=group_cols)
    inactive["became_active"] = inactive["activity_to"] == 1
    out = (
        inactive.groupby(group_cols, dropna=False)
        .agg(
            n_inactive_opportunities=("became_active", "size"),
            n_inactive_to_active=("became_active", "sum"),
        )
        .reset_index()
    )
    out["n_inactive_to_active"] = out["n_inactive_to_active"].astype(int)
    out["inactive_to_active_rate_per_frame"] = (
        out["n_inactive_to_active"] / out["n_inactive_opportunities"]
    )
    out["inactive_to_active_rate_per_second"] = (
        out["inactive_to_active_rate_per_frame"] * float(fps)
    )
    out["transition_label"] = "0_to_1"
    return out


def merge_context_sizes(
    summary: pd.DataFrame,
    date_sizes: pd.DataFrame,
    video_sizes: pd.DataFrame,
    colony_sizes: pd.DataFrame,
) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    if "Date" in out and not date_sizes.empty:
        out = out.merge(date_sizes, on="Date", how="left")
    if "Date" in out and not colony_sizes.empty:
        out = out.merge(colony_sizes, on="Date", how="left")
    if all(col in out for col in VIDEO_COLS) and not video_sizes.empty:
        out = out.merge(video_sizes, on=VIDEO_COLS, how="left")
    return out


def write_for_plot_views(out_dir: Path, collapsed: pd.DataFrame, inactive_date: pd.DataFrame) -> None:
    if not collapsed.empty:
        plot_ready = collapsed.rename(columns={NEST_STATUS_COL: "location_zone"})
        plot_ready.to_csv(out_dir / "activity_transition_rates_by_nest_threshold_contact_for_plot.csv", index=False)
    if not inactive_date.empty:
        plot_ready = inactive_date.rename(columns={NEST_STATUS_COL: "location_zone"})
        plot_ready.to_csv(
            out_dir / "inactive_to_active_rates_by_date_nest_threshold_contact_for_plot.csv",
            index=False,
        )


def main() -> int:
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    frame_index_path = results_root / "frame_level_file_index.csv"
    if not frame_index_path.exists():
        raise SystemExit(f"Missing frame index: {frame_index_path}")

    upstream_settings = read_json(results_root / "run_metadata.json").get("settings", {})
    excluded_ids = set(args.exclude_bee_ids) | set(upstream_settings.get("exclude_bee_ids", []))
    fps = fps_from_metadata(results_root, args.fps)
    colony_size_file = args.colony_size_file or default_colony_size_file(results_root)
    colony_sizes = load_colony_sizes(colony_size_file)
    out_dir = output_dir_for(results_root, sorted(excluded_ids), args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_index = pd.read_csv(frame_index_path)
    frame_index = filter_frame_index_by_date(frame_index, args.start_date, args.end_date)
    if frame_index.empty:
        raise SystemExit("No frame-level date files remain after applying the requested date filter.")
    contact_index = load_contact_index(results_root)
    metadata = read_json(results_root / "run_metadata.json")
    nest_on_distance_px = metadata.get("settings", {}).get("nest_on_distance_px")

    date_transition_parts = []
    video_transition_parts = []
    inactive_date_parts = []
    inactive_video_parts = []
    date_sizes_parts = []
    video_sizes_parts = []
    audit_rows = []
    excluded_detection_parts = []

    for row in tqdm(frame_index.itertuples(index=False), total=len(frame_index), desc="Dates"):
        date = str(row.Date)
        frame_path = resolve_index_path(results_root, row.path)
        frame_level = pd.read_parquet(frame_path)
        excluded_detection_parts.append(excluded_tag_detection_audit(frame_level, excluded_ids))
        filtered, audit = recompute_contacts(frame_level, contact_index.get(date), excluded_ids)
        audit_rows.append({"Date": date, **audit})

        date_sizes, video_sizes = observed_sizes(filtered)
        date_sizes_parts.append(date_sizes)
        video_sizes_parts.append(video_sizes)

        events = transition_events(filtered)
        date_transition_parts.append(
            summarize_transition_counts(events, ["Date", NEST_STATUS_COL, "social_contact"])
        )
        video_transition_parts.append(
            summarize_transition_counts(events, [*VIDEO_COLS, NEST_STATUS_COL, "social_contact"])
        )
        inactive_date_parts.append(
            summarize_inactive_to_active(events, ["Date", NEST_STATUS_COL, "social_contact"], fps)
        )
        inactive_video_parts.append(
            summarize_inactive_to_active(events, [*VIDEO_COLS, NEST_STATUS_COL, "social_contact"], fps)
        )

    date_transitions = (
        add_transition_rates(pd.concat(date_transition_parts, ignore_index=True), fps)
        if date_transition_parts
        else pd.DataFrame()
    )
    video_transitions = (
        add_transition_rates(pd.concat(video_transition_parts, ignore_index=True), fps)
        if video_transition_parts
        else pd.DataFrame()
    )
    collapsed_transitions = collapse_transition_rates(
        date_transitions,
        [NEST_STATUS_COL, "social_contact"],
        fps,
    )
    inactive_date = pd.concat(inactive_date_parts, ignore_index=True) if inactive_date_parts else pd.DataFrame()
    inactive_video = pd.concat(inactive_video_parts, ignore_index=True) if inactive_video_parts else pd.DataFrame()

    date_sizes = pd.concat(date_sizes_parts, ignore_index=True) if date_sizes_parts else pd.DataFrame()
    if not date_sizes.empty:
        date_sizes = (
            date_sizes.groupby("Date", dropna=False)["observed_colony_size_nonexcluded"]
            .max()
            .reset_index()
        )
    video_sizes = pd.concat(video_sizes_parts, ignore_index=True) if video_sizes_parts else pd.DataFrame()

    nonempty_audits = [part for part in excluded_detection_parts if not part.empty]
    excluded_detection_audit = pd.concat(nonempty_audits, ignore_index=True) if nonempty_audits else pd.DataFrame(columns=["Date", "bee_ID", "frame_rows", "detected_rows", "videos_with_tag_column"])

    date_transitions = merge_context_sizes(date_transitions, date_sizes, video_sizes, colony_sizes)
    video_transitions = merge_context_sizes(video_transitions, date_sizes, video_sizes, colony_sizes)
    inactive_date = merge_context_sizes(inactive_date, date_sizes, video_sizes, colony_sizes)
    inactive_video = merge_context_sizes(inactive_video, date_sizes, video_sizes, colony_sizes)

    off_nest_transitions = collapsed_transitions.loc[
        collapsed_transitions[NEST_STATUS_COL] == "off_nest_threshold"
    ].copy()
    off_nest_inactive_date = inactive_date.loc[
        inactive_date[NEST_STATUS_COL] == "off_nest_threshold"
    ].copy()

    date_transitions.to_csv(out_dir / "activity_transition_rates_by_date_nest_threshold_contact.csv", index=False)
    video_transitions.to_csv(out_dir / "activity_transition_rates_by_video_nest_threshold_contact.csv", index=False)
    collapsed_transitions.to_csv(out_dir / "activity_transition_rates_by_nest_threshold_contact.csv", index=False)
    inactive_date.to_csv(out_dir / "inactive_to_active_rates_by_date_nest_threshold_contact.csv", index=False)
    inactive_video.to_csv(out_dir / "inactive_to_active_rates_by_video_nest_threshold_contact.csv", index=False)
    off_nest_transitions.to_csv(out_dir / "off_nest_activity_transition_rates_by_contact.csv", index=False)
    off_nest_inactive_date.to_csv(out_dir / "off_nest_inactive_to_active_rates_by_date_contact.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(out_dir / "excluded_tag_contact_audit_by_date.csv", index=False)
    excluded_detection_audit.to_csv(out_dir / "excluded_tag_detection_audit_by_date.csv", index=False)
    write_for_plot_views(out_dir, collapsed_transitions, inactive_date)

    summary_metadata = {
        "source_results_root": str(results_root),
        "output_dir": str(out_dir),
        "fps": fps,
        "excluded_bee_ids": sorted(excluded_ids),
        "colony_size_file": str(colony_size_file) if colony_size_file else None,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "nest_status_column": NEST_STATUS_COL,
        "nest_status_definition": {
            "on_nest_threshold": (
                "within_nest_perimeter_threshold is True; distance_to_nest_perimeter_px "
                "is <= nest_on_distance_px"
            ),
            "off_nest_threshold": (
                "within_nest_perimeter_threshold is False; distance_to_nest_perimeter_px "
                "is > nest_on_distance_px"
            ),
            "unknown_nest_threshold": "nest threshold status could not be measured for that frame",
        },
        "nest_on_distance_px": nest_on_distance_px,
        "outputs": [
            "activity_transition_rates_by_date_nest_threshold_contact.csv",
            "activity_transition_rates_by_video_nest_threshold_contact.csv",
            "activity_transition_rates_by_nest_threshold_contact.csv",
            "inactive_to_active_rates_by_date_nest_threshold_contact.csv",
            "inactive_to_active_rates_by_video_nest_threshold_contact.csv",
            "off_nest_activity_transition_rates_by_contact.csv",
            "off_nest_inactive_to_active_rates_by_date_contact.csv",
            "activity_transition_rates_by_nest_threshold_contact_for_plot.csv",
            "inactive_to_active_rates_by_date_nest_threshold_contact_for_plot.csv",
        ],
    }
    (out_dir / "nest_transition_summary_metadata.json").write_text(
        json.dumps(summary_metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Excluded bee IDs: {sorted(excluded_ids)}")
    if args.start_date or args.end_date:
        print(f"Date filter: {args.start_date or 'first'} through {args.end_date or 'last'}")
    if colony_size_file:
        print(f"Colony size file: {colony_size_file}")
    print(f"Nest threshold distance: {nest_on_distance_px} px")
    print(f"Date transition rows: {len(date_transitions)}")
    print(f"Inactive-to-active date rows: {len(inactive_date)}")
    print(f"Off-nest inactive-to-active date rows: {len(off_nest_inactive_date)}")
    print(f"Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
