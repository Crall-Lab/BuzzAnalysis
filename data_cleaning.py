#!/usr/bin/env python

import pandas as pd
import numpy as np
import math

#August added back into data_cleaning.py on May 14th, 2025
#Remove tag detections that jump over a threshold number of pixels from one frame to the very next frame
#NOTE: this is not yet robust and its effect needs to be tested - the following scenario presents an issue for the current code:
#Tag 20 is detected in frame 5, 6, and 7
#The first detection, upon review, is the false detection - it is across the nest from where bee 20 actually is.
#The next detection jumps back to the correct position. But which detection actually gets dropped?
#My current understanding upon review and without testing is that the true tag detection would be dropped, which is incorrect.
#I think the third detection would be fine in this case.
#BUT WAIT: in the scenario where bee 20 is tracked in 5,6,7: if detection 6 is the false detection, the diff row between 6 and 7
#would also be flagged, and would detection 7 be removed? Needs testing! 
import math
import pandas as pd
import os

def remove_jumps(df, args):
    """
    Flags suspicious jumps in ArUco tag tracking data and logs jump rows + neighbors.

    Args:
        interpolated_df (pd.DataFrame): tracking data with columns ['ID', 'frame', 'centroidX', 'centroidY']
        jump_thresh (float): pixel threshold for detecting jumps
        log_path (str): path to append the jump log CSV file
        video_id (str): identifier for the current video

    Returns:
        pd.DataFrame: DataFrame with 'flagged_as_jump' column added
    """
    cleaned_df = df.copy()
    cleaned_df['flagged_as_jump'] = False
    jump_thresh = args.remove_jumps
    log_path = f"{os.path.dirname(args.source)}/{os.path.basename(args.source)}_jump_log.csv"
    print(log_path)

    if len(df['filename'].unique()) == 1:
        video_id = df.loc[0, 'filename']
        print(video_id)


    log_entries = []

    for bee_id in cleaned_df['ID'].unique():
        bee_df = cleaned_df[cleaned_df['ID'] == bee_id].sort_values('frame')
        positions = bee_df[['centroidX', 'centroidY']].values
        frames = bee_df['frame'].values
        indices = bee_df.index.values

        for i in range(1, len(positions) - 1):
            frame_prev = frames[i - 1]
            frame_curr = frames[i]
            frame_next = frames[i + 1]

            if frame_curr - frame_prev == 1 and frame_next - frame_curr == 1:
                prev = positions[i - 1]
                curr = positions[i]
                next = positions[i + 1]

                dist_prev = math.dist(prev, curr)
                dist_next = math.dist(curr, next)

                if dist_prev > jump_thresh and dist_next > jump_thresh:
                    jump_index = indices[i]
                    prev_index = indices[i - 1]
                    next_index = indices[i + 1]

                    cleaned_df.loc[jump_index, 'flagged_as_jump'] = True

                    # Add log entries
                    for idx, label in zip([prev_index, jump_index, next_index], ['neighbor', 'jump', 'neighbor']):
                        row = cleaned_df.loc[idx].copy()
                        row['video'] = video_id
                        row['label'] = label
                        log_entries.append(row)

    # Append to CSV log
    if log_entries:
        log_df = pd.DataFrame(log_entries)
        log_columns = ['video', 'ID', 'frame', 'centroidX', 'centroidY', 'label']
        log_df = log_df[log_columns]

        write_header = not os.path.exists(log_path)
        print(f"write_header: {write_header}")
        log_df.to_csv(log_path, mode='a', header=write_header, index=False)
    else:
        print("No jumps detected -> no jump log written")
    return cleaned_df

def summarize_jump_log(args):
    """
    Summarize total jump detections per bee across all videos.

    Args:
        log_path (str): path to the CSV log file

    Prints:
        Total jump counts per bee ID and optional per video.
    """
    log_path = f"{os.path.dirname(args.source)}/{os.path.basename(args.source)}_jump_log.csv"

    if not os.path.exists(log_path):
        print("No log file found.")
        return

    log_df = pd.read_csv(log_path)

    summary = (
        log_df[log_df['label'] == 'jump']
        .groupby('ID')
        .size()
        .reset_index(name='n_jumps')
        .sort_values('n_jumps', ascending=False)
    )

    print("🐝 Potential Tag Jump Summary by Bee ID:")
    print(summary.to_string(index=False))

#check for multiples of the same tag in each frame
def return_duplicate_bees(df):
    df.drop_duplicates(inplace=True)  # Drop completely duplicate rows
    try:
        #for the duplicated function based on specific columns, keep=False makes it so that both duplicates are marked True
        df['in_frame_duplicate'] = df.duplicated(['filename', 'ID', 'frame', 'colony number'], keep=False)
        if True in df['in_frame_duplicate'].values:
            print('Yes, there are duplicate tag readings in the same frame! They’ve been marked True in the duplicates column.')
            return df, 0
        else:
            print('There aren’t any duplicates in this dataframe!')
            return df, 1
    except Exception as e:
        print(f'''An error occurred in return_duplicate_bees(): {str(e)}
                 Ensure the DataFrame has the required columns: 'filename', 'ID', 'frame', 'colony number'.''')
        return df, 1  # Return original DataFrame to avoid breaking downstream logic


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
    """
    Resolves duplicate detections of the same bee ID within a single frame based on spatial proximity
    to known positions in neighboring frames. Keeps the most plausible tag and optionally flags or
    drops unresolved duplicates.

    Parameters:
    df (DataFrame): The tracking data with potential duplicates.
    return_val (int): Returned value from return_duplicate_bees(), 0 if duplicates exist.
    drop_unresolvable (bool): Whether to drop duplicate rows that couldn't be confidently resolved.

    Returns:
    DataFrame: A cleaned DataFrame with resolved duplicates removed and the best candidate retained.
    """
    df = df.copy()  # Work on a copy to avoid modifying original data
    df.drop_duplicates(inplace=True)  # Drop any fully duplicated rows

    # Sanity check to make sure duplicates have already been identified
    if 'in_frame_duplicate' not in df.columns:
        print("Hey, have you run the return_duplicate_bees() function? I'm not seeing a duplicate column in this dataframe.")
        return df

    # Add helper columns to track which rows were part of a duplicate set and what happened to them
    df['og_duplicate'] = False
    df['unresolvable_duplicate'] = False

    if return_val == 0:
        # Subset to rows marked as duplicates
        duplicates = df[df['duplicate'] == True]

        # Create a table of unique (video, colony, bee ID, frame) combinations with duplicates
        dupe_keys = duplicates[['video path', 'col #', 'bee ID', 'frame number']].drop_duplicates()

        # Loop through each unique duplicated instance
        for _, row in dupe_keys.iterrows():
            vid = row['video path']
            col = row['col #']
            bee = row['bee ID']
            frame = row['frame number']

            # Get all the duplicated rows for this (video, colony, bee ID, frame)
            specific_duplicates = duplicates[
                (duplicates['video path'] == vid) &
                (duplicates['col #'] == col) &
                (duplicates['bee ID'] == bee) &
                (duplicates['frame number'] == frame)
            ]

            # Find other positions of the same bee in other frames (same video)
            nearest_position_v1 = df[
                (df['video path'] == vid) &
                (df['col #'] == col) &
                (df['bee ID'] == bee) &
                (df['frame number'] != frame)
            ]

            # If no known positions exist in other frames, we can't resolve this duplicate
            if nearest_position_v1.empty:
                df.loc[specific_duplicates.index, 'unresolvable_duplicate'] = True
                continue

            # Find the position in another frame that is temporally closest to the duplicate frame
            nearest_position_v2 = nearest_position_v1.iloc[
                (nearest_position_v1['frame number'] - frame).abs().argsort()[:1]
            ]

            # Skip resolution if the nearest frame is too far away to trust
            if nearest_position_v2.empty or abs(nearest_position_v2['frame number'].values[0] - frame) > 16:
                df.loc[specific_duplicates.index, 'unresolvable_duplicate'] = True
                continue

            # Call modular function to find best candidate detection among the duplicates
            idx_to_keep, tag_to_keep = resolve_duplicate_by_proximity(
                specific_duplicates, nearest_position_v2.iloc[0]
            )

            # Drop all other candidates in the same frame with same ID
            drop_idxs = df[
                (df['video path'] == vid) &
                (df['col #'] == col) &
                (df['bee ID'] == bee) &
                (df['frame number'] == frame) &
                ((df['centroidX'] != tag_to_keep['centroidX']) |
                 (df['centroidY'] != tag_to_keep['centroidY']))
            ].index
            df.drop(index=drop_idxs, inplace=True)

            # Mark the kept tag as a resolved original duplicate
            good_idx = df[
                (df['video path'] == vid) &
                (df['col #'] == col) &
                (df['bee ID'] == bee) &
                (df['frame number'] == frame) &
                (df['centroidX'] == tag_to_keep['centroidX']) &
                (df['centroidY'] == tag_to_keep['centroidY'])
            ].index
            df.loc[good_idx, 'duplicate'] = False
            df.loc[good_idx, 'og_duplicate'] = True

        # Optionally remove any unresolved duplicates
        if drop_unresolvable:
            df.drop(df[df['duplicate'] == True].index, inplace=True)

    elif return_val == 1 and isinstance(df, pd.DataFrame):
        # If no duplicates existed, still ensure tracking columns exist
        df['og_duplicate'] = False
        df['unresolvable_duplicate'] = False

    return df


# Updated function to interpolate missing frames only if the gap between them is less than or equal to max_frame_gap
# Updated on May 14th by August to add an interpolation column marking 0 as not an interpolated row, and 1 as yes interpolated
def interpolate(df, max_seconds_gap, actual_frames_per_second):
    max_frame_gap = int(max_seconds_gap * actual_frames_per_second)
    print(f"Max frame gap based on --max-interp-sec and --real-fps: {max_frame_gap}")

    #drop rows flagged as jumps to avoid interpolating over them
    df = df[df['flagged_as_jump'] != True]

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
                    num_frames_to_interpolate = next_row['frame_diff'] - 1
                    for n in range(1, num_frames_to_interpolate + 1):
                        interp_row = row.copy()
                        ratio = n / next_row['frame_diff']
                        # Interpolate the position columns
                        for col in ['centroidX', 'centroidY', 'frontX', 'frontY']:
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

    # Concatenate all groups
    interpolated_df = pd.concat(interpolated_dfs, ignore_index=True)

    # Sort for clarity
    interpolated_df.sort_values(by=['ID', 'frame'], inplace=True)

    return interpolated_df


def main():
	print("I am a python module, I am not run by myself. I just contain functions that are imported by other scripts to use!")
	
if __name__ == '__main__':
	
	main()
