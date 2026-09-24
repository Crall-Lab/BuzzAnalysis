#!/usr/bin/env python3
"""Summarize inactive-to-active transitions from preliminary frame-level outputs.

This is intended for quick downstream summaries after
preliminary_newtracks_analysis.py has written frame_level_by_date and
contact_pairs_by_date outputs. It can exclude known-problem tags and recompute
contact_count/social_contact from the sparse contact-pair files so excluded
tags do not contribute to other bees' social-contact context.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm


CONTEXT_COLS = [
    "location_zone",
    "social_contact",
    "on_brood",
    "on_pollen",
    "on_nectar",
    "on_wax",
    "on_any_small_object",
]

VIDEO_COLS = ["Date", "Time", "datetime", "source_file"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build inactive-to-active summaries from preliminary BuzzAnalysis outputs."
    )
    parser.add_argument("results_root", type=Path, help="Preliminary results folder.")
    parser.add_argument(
        "--exclude-bee-ids",
        nargs="*",
        type=int,
        default=[],
        help="Bee IDs to drop before summarizing; contact pairs involving these IDs are also ignored.",
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
        help="Folder for outputs; default is results_root/inactive_to_active_excluding_<ids>.",
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


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_index_path(results_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return results_root / path


def output_dir_for(results_root: Path, excluded_ids: Iterable[int], requested: Path | None) -> Path:
    if requested is not None:
        return requested.expanduser().resolve()
    suffix = "_".join(str(bee_id) for bee_id in excluded_ids) or "none"
    return (results_root / f"inactive_to_active_excluding_{suffix}").resolve()


def fps_from_metadata(results_root: Path, requested_fps: float | None) -> float:
    metadata = read_json(results_root / "run_metadata.json")
    settings = metadata.get("settings", {})
    fps = float(requested_fps if requested_fps is not None else settings.get("fps", metadata.get("fps", 4.5)))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    return fps


def default_colony_size_file(results_root: Path) -> Path | None:
    colony = read_json(results_root / "run_metadata.json").get("colony_number")
    if colony is None:
        return None
    candidates = [results_root.parent / f"size-over-time-colony-{colony}.csv",
                  results_root / f"size-over-time-colony-{colony}.csv"]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_colony_sizes(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    data = pd.read_csv(path)
    lowered = {str(col).strip().lower(): col for col in data.columns}
    date_col = lowered.get("date")
    size_col = (
        lowered.get("colony.size")
        or lowered.get("colony_size")
        or lowered.get("colony size")
        or lowered.get("size")
    )
    if date_col is None or size_col is None:
        raise ValueError(
            f"{path} must contain a date column and a colony size column; found {list(data.columns)}"
        )
    out = data[[date_col, size_col]].copy()
    out.columns = ["Date", "true_colony_size"]
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out = out.dropna(subset=["Date", "true_colony_size"])
    out["true_colony_size"] = out["true_colony_size"].astype(int)
    return out.drop_duplicates(subset=["Date"], keep="last")


def load_contact_index(results_root: Path) -> dict[str, Path]:
    index_path = results_root / "contact_pairs_file_index.csv"
    if not index_path.exists():
        return {}
    try:
        index = pd.read_csv(index_path)
    except pd.errors.EmptyDataError:
        return {}
    if "Date" not in index or "path" not in index:
        return {}
    return {
        str(row.Date): resolve_index_path(results_root, row.path)
        for row in index.itertuples(index=False)
        if isinstance(row.path, str)
    }


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


def recompute_contacts(
    frame_level: pd.DataFrame,
    contact_path: Path | None,
    excluded_ids: set[int],
) -> tuple[pd.DataFrame, dict[str, int]]:
    excluded_ids = set(excluded_ids)
    if "excluded_from_analysis" in frame_level:
        upstream = frame_level["excluded_from_analysis"].fillna(False).astype(bool)
        excluded_ids.update(frame_level.loc[upstream, "bee_ID"].astype(int))
    work = frame_level.loc[~frame_level["bee_ID"].isin(excluded_ids)].copy().reset_index(drop=True)

    audit = {
        "contact_pairs_original": 0,
        "contact_pairs_kept": 0,
        "contact_pairs_removed_for_excluded_tags": 0,
    }

    if contact_path is not None and not contact_path.exists():
        raise FileNotFoundError(f"Missing contact-pair file: {contact_path}")
    if contact_path is None:
        # The producer omits pair files only when no contacts occurred. Preserve
        # unknown counts, and reject inconsistent indexes instead of inventing zeros.
        if frame_level["contact_count"].fillna(0).gt(0).any():
            raise ValueError("Missing contact-pair index for data containing contacts")
        return work, audit

    pairs = pd.read_parquet(contact_path)
    audit["contact_pairs_original"] = int(len(pairs))
    if pairs.empty:
        detected = work["centroidX"].notna() & work["centroidY"].notna()
        work["contact_count"] = np.where(detected, 0, np.nan)
        work["social_contact"] = pd.Series(work["contact_count"] > 0, dtype="boolean").where(detected, pd.NA)
        return work, audit

    required = {"source_file", "frame", "bee_ID_1", "bee_ID_2"}
    if not required.issubset(pairs.columns):
        raise ValueError(f"Contact-pair file missing columns: {sorted(required - set(pairs.columns))}")

    has_excluded = pairs["bee_ID_1"].isin(excluded_ids) | pairs["bee_ID_2"].isin(excluded_ids)
    audit["contact_pairs_removed_for_excluded_tags"] = int(has_excluded.sum())
    pairs = pairs.loc[~has_excluded].copy()
    audit["contact_pairs_kept"] = int(len(pairs))

    count_parts = []
    for col in ("bee_ID_1", "bee_ID_2"):
        part = (
            pairs.groupby(["source_file", "frame", col], dropna=False)
            .size()
            .reset_index(name="pair_count")
            .rename(columns={col: "bee_ID"})
        )
        count_parts.append(part)

    if count_parts:
        counts = pd.concat(count_parts, ignore_index=True)
        counts = (
            counts.groupby(["source_file", "frame", "bee_ID"], dropna=False)["pair_count"]
            .sum()
            .reset_index()
        )
        work = work.merge(counts, on=["source_file", "frame", "bee_ID"], how="left")
        contact_count = work["pair_count"].fillna(0).astype(float)
        work = work.drop(columns=["pair_count"])
    else:
        contact_count = pd.Series(0.0, index=work.index)

    detected = work["centroidX"].notna() & work["centroidY"].notna()
    work["contact_count"] = np.where(detected, contact_count, np.nan)
    social_contact = pd.Series(pd.NA, index=work.index, dtype="boolean")
    social_contact.loc[detected] = work.loc[detected, "contact_count"] > 0
    work["social_contact"] = social_contact
    return work, audit


def normalize_context_columns(events: pd.DataFrame) -> pd.DataFrame:
    def normalize_boolish(value: object) -> str:
        if pd.isna(value):
            return "unknown"
        if isinstance(value, (bool, np.bool_)):
            return "True" if value else "False"
        text = str(value).strip().lower()
        if text in {"true", "1", "yes"}:
            return "True"
        if text in {"false", "0", "no"}:
            return "False"
        return "unknown"

    for col in CONTEXT_COLS:
        if col not in events:
            events[col] = "unknown"
            continue
        if col == "location_zone":
            events[col] = events[col].fillna("unknown").astype(str)
        else:
            events[col] = events[col].map(normalize_boolish)
    return events


def inactive_transition_events(frame_level: pd.DataFrame) -> pd.DataFrame:
    if frame_level.empty:
        return pd.DataFrame()

    work = frame_level.sort_values(["source_file", "bee_ID", "frame"]).copy()
    grouped = work.groupby(["source_file", "bee_ID"], sort=False)
    work["next_activity"] = grouped["activity"].shift(-1)
    work["next_frame"] = grouped["frame"].shift(-1)
    valid = (
        work["activity"].isin([0, 1])
        & work["next_activity"].isin([0, 1])
        & ((work["next_frame"] - work["frame"]) == 1)
    )
    events = work.loc[valid].copy()
    events = events.loc[events["activity"].astype(int) == 0].copy()
    if events.empty:
        return pd.DataFrame()

    events["became_active"] = events["next_activity"].astype(int) == 1
    return normalize_context_columns(events)


def summarize_events(events: pd.DataFrame, group_cols: list[str], fps: float) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=group_cols)
    summary = (
        events.groupby(group_cols, dropna=False)
        .agg(
            n_inactive_opportunities=("became_active", "size"),
            n_inactive_to_active=("became_active", "sum"),
        )
        .reset_index()
    )
    summary["n_inactive_to_active"] = summary["n_inactive_to_active"].astype(int)
    summary["inactive_to_active_rate_per_frame"] = (
        summary["n_inactive_to_active"] / summary["n_inactive_opportunities"]
    )
    summary["inactive_to_active_rate_per_second"] = summary["inactive_to_active_rate_per_frame"] * float(fps)
    summary["transition_label"] = "0_to_1"
    return summary


def observed_sizes(frame_level: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detected = frame_level.loc[frame_level["centroidX"].notna() & frame_level["centroidY"].notna()]
    date_sizes = (
        detected.groupby("Date", dropna=False)["bee_ID"]
        .nunique()
        .reset_index(name="observed_colony_size_nonexcluded")
    )
    video_sizes = (
        detected.groupby(VIDEO_COLS, dropna=False)["bee_ID"]
        .nunique()
        .reset_index(name="observed_bees_in_video_nonexcluded")
    )
    return date_sizes, video_sizes


def excluded_tag_detection_audit(frame_level: pd.DataFrame, excluded_ids: set[int]) -> pd.DataFrame:
    if frame_level.empty or not excluded_ids:
        return pd.DataFrame()
    sub = frame_level.loc[frame_level["bee_ID"].isin(excluded_ids)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["detected"] = sub["centroidX"].notna() & sub["centroidY"].notna()
    return (
        sub.groupby(["Date", "bee_ID"], dropna=False)
        .agg(
            frame_rows=("bee_ID", "size"),
            detected_rows=("detected", "sum"),
            videos_with_tag_column=("source_file", "nunique"),
        )
        .reset_index()
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

    date_summaries = []
    video_summaries = []
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

        events = inactive_transition_events(filtered)
        date_summaries.append(summarize_events(events, ["Date", *CONTEXT_COLS], fps))
        video_summaries.append(summarize_events(events, [*VIDEO_COLS, *CONTEXT_COLS], fps))

    date_summary = pd.concat(date_summaries, ignore_index=True) if date_summaries else pd.DataFrame()
    video_summary = pd.concat(video_summaries, ignore_index=True) if video_summaries else pd.DataFrame()
    date_sizes = pd.concat(date_sizes_parts, ignore_index=True) if date_sizes_parts else pd.DataFrame()
    video_sizes = pd.concat(video_sizes_parts, ignore_index=True) if video_sizes_parts else pd.DataFrame()
    nonempty_audits = [part for part in excluded_detection_parts if not part.empty]
    excluded_detection_audit = pd.concat(nonempty_audits, ignore_index=True) if nonempty_audits else pd.DataFrame(columns=["Date", "bee_ID", "frame_rows", "detected_rows", "videos_with_tag_column"])

    if not date_sizes.empty:
        date_sizes = (
            date_sizes.groupby("Date", dropna=False)["observed_colony_size_nonexcluded"]
            .max()
            .reset_index()
        )
        date_summary = date_summary.merge(date_sizes, on="Date", how="left")
    if not colony_sizes.empty:
        date_summary = date_summary.merge(colony_sizes, on="Date", how="left")
        video_summary = video_summary.merge(colony_sizes, on="Date", how="left")
    if not video_sizes.empty:
        video_summary = video_summary.merge(video_sizes, on=VIDEO_COLS, how="left")
    if not date_sizes.empty and not video_summary.empty:
        video_summary = video_summary.merge(date_sizes, on="Date", how="left")

    date_summary.to_csv(out_dir / "inactive_to_active_rates_by_date_location_contact.csv", index=False)
    video_summary.to_csv(out_dir / "inactive_to_active_rates_by_video_location_contact.csv", index=False)
    date_sizes.to_csv(out_dir / "observed_colony_size_by_date.csv", index=False)
    video_sizes.to_csv(out_dir / "observed_colony_size_by_video.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(out_dir / "excluded_tag_contact_audit_by_date.csv", index=False)
    excluded_detection_audit.to_csv(out_dir / "excluded_tag_detection_audit_by_date.csv", index=False)

    metadata = {
        "source_results_root": str(results_root),
        "output_dir": str(out_dir),
        "fps": fps,
        "excluded_bee_ids": sorted(excluded_ids),
        "colony_size_file": str(colony_size_file) if colony_size_file else None,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "context_columns": CONTEXT_COLS,
        "outputs": [
            "inactive_to_active_rates_by_date_location_contact.csv",
            "inactive_to_active_rates_by_video_location_contact.csv",
            "observed_colony_size_by_date.csv",
            "observed_colony_size_by_video.csv",
            "excluded_tag_contact_audit_by_date.csv",
            "excluded_tag_detection_audit_by_date.csv",
        ],
    }
    (out_dir / "inactive_to_active_summary_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Excluded bee IDs: {sorted(excluded_ids)}")
    if args.start_date or args.end_date:
        print(f"Date filter: {args.start_date or 'first'} through {args.end_date or 'last'}")
    if colony_size_file:
        print(f"Colony size file: {colony_size_file}")
    print(f"Date summary rows: {len(date_summary)}")
    print(f"Video summary rows: {len(video_summary)}")
    print(f"Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
