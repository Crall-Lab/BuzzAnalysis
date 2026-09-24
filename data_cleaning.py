#!/usr/bin/env python

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _jump_log_path(args):
    source = Path(args.source).expanduser().resolve()
    return str(source.parent / f"{source.name}_jump_log.csv")


def _speed_between(prev_frame, prev_position, next_frame, next_position):
    frame_gap = int(next_frame) - int(prev_frame)
    if frame_gap <= 0:
        return np.nan
    return math.dist(prev_position, next_position) / frame_gap


def _append_jump_context(
    log_entries,
    cleaned_df,
    *,
    indices,
    labels,
    video_id,
    action,
    reason,
    speed_prev_curr=np.nan,
    speed_curr_next=np.nan,
    speed_prev_next=np.nan,
    transition_speed=np.nan,
    frame_gap=np.nan,
    threshold=np.nan,
):
    for idx, label in zip(indices, labels):
        row = cleaned_df.loc[idx].copy()
        row['video'] = video_id
        row['label'] = label
        row['action'] = action
        row['reason'] = reason
        row['speed_prev_curr_px_per_frame'] = speed_prev_curr
        row['speed_curr_next_px_per_frame'] = speed_curr_next
        row['speed_prev_next_px_per_frame'] = speed_prev_next
        row['transition_speed_px_per_frame'] = transition_speed
        row['frame_gap'] = frame_gap
        row['max_speed_px_per_frame'] = threshold
        log_entries.append(row)


def remove_jumps(df, args, filename):
    """
    Remove clear isolated ArUco tag-read spikes and log ambiguous high-speed transitions.

    A point is removed only when it is a jump-out-and-back spike:
      previous -> current exceeds the threshold,
      current -> next exceeds the threshold,
      previous -> next does not exceed the threshold.

    One-way high-speed transitions are logged but retained. The downstream
    analysis max-speed filter should mark those transitions as unknown.

    Args:
        df (pd.DataFrame): tracking data with columns ['ID', 'frame', 'centroidX', 'centroidY']
        args.remove_jumps (float): maximum plausible speed in pixels per frame
        video_id (str): identifier for the current video

    Returns:
        pd.DataFrame: DataFrame with clear isolated spike rows removed.
    """
    cleaned_df = df.copy()
    cleaned_df['flagged_as_jump'] = False
    jump_thresh = float(args.remove_jumps)
    log_path = _jump_log_path(args)
    print(log_path)

    video_id = filename
    print(video_id)

    log_entries = []

    for bee_id in cleaned_df['ID'].unique():
        bee_df = cleaned_df[cleaned_df['ID'] == bee_id].sort_values('frame')
        positions = bee_df[['centroidX', 'centroidY']].values
        frames = bee_df['frame'].values
        indices = bee_df.index.values

        flagged_indices = set()

        for i in range(1, len(positions) - 1):
            frame_prev = frames[i - 1]
            frame_curr = frames[i]
            frame_next = frames[i + 1]

            prev = positions[i - 1]
            curr = positions[i]
            next_pos = positions[i + 1]

            speed_prev_curr = _speed_between(frame_prev, prev, frame_curr, curr)
            speed_curr_next = _speed_between(frame_curr, curr, frame_next, next_pos)
            speed_prev_next = _speed_between(frame_prev, prev, frame_next, next_pos)

            if (
                indices[i - 1] not in flagged_indices
                and speed_prev_curr > jump_thresh
                and speed_curr_next > jump_thresh
                and speed_prev_next <= jump_thresh
            ):
                jump_index = indices[i]
                prev_index = indices[i - 1]
                next_index = indices[i + 1]

                cleaned_df.loc[jump_index, 'flagged_as_jump'] = True
                flagged_indices.add(jump_index)

                _append_jump_context(
                    log_entries,
                    cleaned_df,
                    indices=[prev_index, jump_index, next_index],
                    labels=['neighbor', 'jump', 'neighbor'],
                    video_id=video_id,
                    action='removed_isolated_spike',
                    reason='jump_out_and_back_bridge_plausible',
                    speed_prev_curr=speed_prev_curr,
                    speed_curr_next=speed_curr_next,
                    speed_prev_next=speed_prev_next,
                    threshold=jump_thresh,
                )

        for i in range(1, len(positions)):
            prev_index = indices[i - 1]
            curr_index = indices[i]
            if prev_index in flagged_indices or curr_index in flagged_indices:
                continue

            frame_prev = frames[i - 1]
            frame_curr = frames[i]
            speed = _speed_between(frame_prev, positions[i - 1], frame_curr, positions[i])
            if speed > jump_thresh:
                _append_jump_context(
                    log_entries,
                    cleaned_df,
                    indices=[prev_index, curr_index],
                    labels=['transition_start', 'transition_end'],
                    video_id=video_id,
                    action='kept_ambiguous_high_speed_transition',
                    reason='one_way_high_speed_transition',
                    transition_speed=speed,
                    frame_gap=int(frame_curr) - int(frame_prev),
                    threshold=jump_thresh,
                )

    # Append to CSV log
    if log_entries:
        log_df = pd.DataFrame(log_entries)
        log_columns = [
            'video',
            'ID',
            'frame',
            'centroidX',
            'centroidY',
            'label',
            'action',
            'reason',
            'speed_prev_curr_px_per_frame',
            'speed_curr_next_px_per_frame',
            'speed_prev_next_px_per_frame',
            'transition_speed_px_per_frame',
            'frame_gap',
            'max_speed_px_per_frame',
        ]
        log_df = log_df[log_columns]

        if os.path.exists(log_path):
            previous = pd.read_csv(log_path)
            if previous.columns.tolist() != log_columns:
                if "action" not in previous and "label" in previous:
                    previous["action"] = np.where(previous["label"] == "jump", "removed_isolated_spike", "unknown")
                previous.reindex(columns=log_columns).to_csv(log_path, index=False)
        write_header = not os.path.exists(log_path)
        print(f"write_header: {write_header}")
        log_df.to_csv(log_path, mode='a', header=write_header, index=False)
    else:
        print("No jumps or ambiguous high-speed transitions detected -> no jump log written")

    jump_mask = cleaned_df['flagged_as_jump']
    n_removed = int(jump_mask.sum())
    if n_removed:
        cleaned_df = cleaned_df.loc[~jump_mask].copy()
        print(f"Removed {n_removed} rows flagged as jumps")

    return cleaned_df.drop(columns=['flagged_as_jump'])

def summarize_jump_log(args):
    """
    Summarize removed spike detections and retained ambiguous transitions.

    Args:
        log_path (str): path to the CSV log file

    Prints:
        Total jump counts per bee ID and optional per video.
    """
    log_path = _jump_log_path(args)

    if not os.path.exists(log_path):
        print("No log file found.")
        return

    log_df = pd.read_csv(log_path)
    if 'action' not in log_df.columns:
        log_df['action'] = np.where(log_df['label'] == 'jump', 'removed_isolated_spike', 'unknown')

    removed = (
        log_df[(log_df['action'] == 'removed_isolated_spike') & (log_df['label'] == 'jump')]
        .groupby('ID')
        .size()
        .reset_index(name='n_removed_spikes')
        .sort_values('n_removed_spikes', ascending=False)
    )

    ambiguous = (
        log_df[
            (log_df['action'] == 'kept_ambiguous_high_speed_transition')
            & (log_df['label'] == 'transition_end')
        ]
        .groupby('ID')
        .size()
        .reset_index(name='n_ambiguous_high_speed_transitions')
        .sort_values('n_ambiguous_high_speed_transitions', ascending=False)
    )

    if removed.empty:
        print("No isolated spike points were removed.")
    else:
        print("Removed isolated spike points by bee ID:")
        print(removed.to_string(index=False))

    if ambiguous.empty:
        print("No ambiguous one-way high-speed transitions were logged.")
    else:
        print("Ambiguous one-way high-speed transitions kept for analysis-time filtering:")
        print(ambiguous.to_string(index=False))

def _duplicate_keys(df):
    required = {"ID", "frame", "centroidX", "centroidY"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Tracking data missing required columns: {sorted(missing)}")
    return [col for col in ("filename", "colony number", "ID", "frame") if col in df]


def return_duplicate_bees(df):
    """Flag repeated IDs within a frame; video/colony metadata is optional."""
    keys = _duplicate_keys(df)
    df = df.drop_duplicates().copy().reset_index(drop=True)
    df["in_frame_duplicate"] = df.duplicated(keys, keep=False)
    return df, int(not df["in_frame_duplicate"].any())


#Helper function that runs inside of the drop_duplicates_clean function (below)
def resolve_duplicate_by_proximity(duplicate_rows, nearest_row):
    """
    Resolve among a group of duplicate tag detections in the same frame by selecting the one
    closest in space to a known position from a nearby frame.

    Parameters:
    duplicate_rows (DataFrame): Rows representing duplicate tag detections in the same frame.
    nearest_row (Series): The nearest known position of the same bee in another frame.

    Returns:
    tuple: (index, row) of the detection that is closest in space to the reference.
    """
    closest_idx = None
    closest_row = None
    min_distance = float('inf')  # Start with an arbitrarily large distance

    # Loop over each duplicate candidate in the current frame
    for idx, row in duplicate_rows.iterrows():
        # Compute Euclidean distance between this candidate and the known nearby position
        dx = row['centroidX'] - nearest_row['centroidX']
        dy = row['centroidY'] - nearest_row['centroidY']
        dist = math.hypot(dx, dy)

        # Keep track of the one closest to the known nearby point
        if dist < min_distance:
            closest_idx = idx
            closest_row = row
            min_distance = dist

    return closest_idx, closest_row


def drop_duplicates_clean(df, return_val, drop_unresolvable=True):
    """Resolve duplicate IDs using the nearest unambiguous observation (within 16 frames)."""
    keys = _duplicate_keys(df)
    df = df.copy()
    df["og_duplicate"] = False
    df["unresolvable_duplicate"] = False
    df["in_frame_duplicate"] = df.duplicated(keys, keep=False)
    context = [key for key in keys if key != "frame"]
    duplicates = df.loc[df["in_frame_duplicate"]]
    drop_indices = []
    for values, candidates in duplicates.groupby(keys, dropna=False, sort=False):
        values = dict(zip(keys, values))
        mask = ~df["in_frame_duplicate"] & (df["frame"] != values["frame"])
        for key in context:
            mask &= df[key].isna() if pd.isna(values[key]) else df[key].eq(values[key])
        neighbors = df.loc[mask].dropna(subset=["centroidX", "centroidY"])
        neighbors = neighbors.loc[(neighbors["frame"] - values["frame"]).abs() <= 16]
        keep = None
        if not neighbors.empty:
            nearest = neighbors.loc[(neighbors["frame"] - values["frame"]).abs().idxmin()]
            keep, _ = resolve_duplicate_by_proximity(candidates, nearest)
        if keep is None:
            df.loc[candidates.index, "unresolvable_duplicate"] = True
            if drop_unresolvable:
                drop_indices.extend(candidates.index)
        else:
            drop_indices.extend(index for index in candidates.index if index != keep)
            df.loc[keep, "in_frame_duplicate"] = False
            df.loc[keep, "og_duplicate"] = True
    return df.drop(index=drop_indices)


# Updated function to interpolate missing frames only if the gap between them is less than or equal to max_frame_gap
# Updated on May 14th by August to add an interpolation column marking 0 as not an interpolated row, and 1 as yes interpolated
def interpolate(df, max_seconds_gap, actual_frames_per_second):
    if not np.isfinite(actual_frames_per_second) or actual_frames_per_second <= 0:
        raise ValueError("actual_frames_per_second must be finite and positive")
    if not np.isfinite(max_seconds_gap) or max_seconds_gap < 0:
        raise ValueError("max_seconds_gap must be finite and nonnegative")
    df = df.copy()
    max_frame_gap = int(max_seconds_gap * actual_frames_per_second)
    print(f"Max frame gap based on --max-interp-sec and --real-fps: {max_frame_gap}")

    # Backward compatibility for dataframes created before jumps were
    # removed inside remove_jumps().
    if 'flagged_as_jump' in df.columns:
        df = df[df['flagged_as_jump'] != True].copy()
        df = df.drop(columns=['flagged_as_jump'])

    # Ensure the data is sorted by ID and frame
    df.sort_values(by=['ID', 'frame'], inplace=True)
    
    # Mark all original rows as not interpolated
    df["interpolated"] = 0

    # Group by bee ID
    grouped = df.groupby('ID')

    # Placeholder for the new DataFrame with interpolated values
    interpolated_dfs = []

    for bee_id, group in grouped:
        # Ensure group is sorted by frame
        group = group.sort_values('frame').copy()
        
        # Calculate the frame difference between consecutive rows
        group['frame_diff'] = group['frame'].diff().fillna(0).astype(int)
        
        # Placeholder list to store the interpolated results for this group
        interpolated_rows = []

        # Iterate over the rows of the group
        for i in range(len(group)):
            row = group.iloc[i]
            interpolated_rows.append(row)

            # Get the next row if it exists
            if i + 1 < len(group):
                next_row = group.iloc[i + 1]
                # If the frame difference is less than or equal to the max frame gap, interpolate
                if 0 < next_row['frame_diff'] <= max_frame_gap:
                    num_frames_to_interpolate = int(next_row['frame_diff']) - 1
                    for n in range(1, num_frames_to_interpolate + 1):
                        interp_row = row.copy()
                        ratio = n / next_row['frame_diff']
                        # Interpolate the position columns
                        for col in [c for c in ('centroidX', 'centroidY', 'frontX', 'frontY') if c in group]:
                            interp_row[col] = row[col] + (next_row[col] - row[col]) * ratio
                        # Set frame number and interpolation flag
                        interp_row['frame'] = int(row['frame'] + n)
                        interp_row["interpolated"] = 1
                        interpolated_rows.append(interp_row)

        # Create a DataFrame from the list of rows
        interpolated_group = pd.DataFrame(interpolated_rows)

        # Drop temporary column
        interpolated_group.drop(columns=['frame_diff'], inplace=True)

        # Append the group to the list
        interpolated_dfs.append(interpolated_group)

    try:
        # Concatenate all groups
        interpolated_df = pd.concat(interpolated_dfs, ignore_index=True)

        # Sort for clarity
        interpolated_df.sort_values(by=['ID', 'frame'], inplace=True)

        return interpolated_df
    
    except ValueError: #I just got a value error saying there arent groups to concatenate?? Not sure what that means in this context, am investigating...

        print(df.shape)
        return df #Try returning the original dataframe instead



def main():
	print("I am a python module, I am not run by myself. I just contain functions that are imported by other scripts to use!")
	
if __name__ == '__main__':
	
	main()
