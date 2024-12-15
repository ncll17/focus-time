import yaml
import pandas as pd
from pandas import Timestamp
import numpy as np
from tqdm import tqdm
from ast import literal_eval


def load_raw_data(transactions_file, app_mappings_file):
    """Load raw data from CSV files."""
    df_day_point = pd.read_csv(transactions_file, index_col=0)
    df_app_to_class = pd.read_csv(app_mappings_file)
    return df_day_point, df_app_to_class


def process_row(row):
    """Process a single row of the dataframe to create exploded format."""
    apps = eval(row["app"])
    durations = eval(row["app_durations"])
    start_times = eval(row["app_start_times"])
    end_times = eval(row["app_end_times"])
    mouse_clicks = eval(row["mouseClicks"])
    keystrokes = eval(row["keystrokes"])
    mic = eval(row["mic"])
    mouse_scroll = eval(row["mouseScroll"])
    camera = eval(row["camera"])

    # Access employeeId from the index (row.name)
    employee_id = row.name

    return pd.DataFrame(
        {
            "app": apps,
            "duration": durations,
            "app_start_time": start_times,
            "app_end_time": end_times,
            "mouseClicks": mouse_clicks,
            "keystrokes": keystrokes,
            "mic": mic,
            "mouseScroll": mouse_scroll,
            "camera": camera,
            "employeeId": [employee_id] * len(apps),
            "workday_start": [row["start_time"]] * len(apps),
            "workday_end": [row["end_time"]] * len(apps),
            "workday_duration": [row["workday_duration"]] * len(apps),
            "hours_until_next_workday": [row["hours_until_next_workday"]] * len(apps),
        }
    )


def create_exploded_df(df_day_point, app_quality_path):
    """Create exploded dataframe from day point data."""
    # Load app quality mapping from yaml file
    with open(app_quality_path, "r") as file:
        app_quality_mapping = yaml.safe_load(file)

    print(f"Columns: {df_day_point.columns}")
    # Create exploded dataframe
    exploded_df = pd.concat(
        [process_row(row) for _, row in df_day_point.iterrows()], ignore_index=True
    )
    exploded_df["duration"] = exploded_df["duration"] * 60  # Convert to seconds

    # Map app quality values, with a default value of 5 for missing apps
    exploded_df["app_quality"] = exploded_df["app"].map(app_quality_mapping).fillna(5)

    return exploded_df


def create_sequences(exploded_df, window_size=64, stride=8):
    """Create sequences using sliding window approach."""
    df_sorted = exploded_df.sort_values(["employeeId", "app_start_time"])
    sequences = []

    for (emp_id,), group in tqdm(
        df_sorted.groupby(["employeeId"]), desc="Creating sequences"
    ):
        apps = group["app"].tolist()
        durations = group["duration"].tolist()

        mouse_clicks = group["mouseClicks"].tolist()
        keystrokes = group["keystrokes"].tolist()
        mic = group["mic"].tolist()
        mouse_scroll = group["mouseScroll"].tolist()
        camera = group["camera"].tolist()
        app_quality = group["app_quality"].tolist()

        for start_idx in range(0, len(apps), stride):
            end_idx = start_idx + window_size

            window_apps = apps[start_idx:end_idx]
            window_durations = durations[start_idx:end_idx]
            window_mouse_clicks = mouse_clicks[start_idx:end_idx]
            window_keystrokes = keystrokes[start_idx:end_idx]
            window_mic = mic[start_idx:end_idx]
            window_mouse_scroll = mouse_scroll[start_idx:end_idx]
            window_camera = camera[start_idx:end_idx]
            window_app_quality = app_quality[start_idx:end_idx]

            if len(window_apps) >= 4:
                sequence = {
                    "apps": window_apps,
                    "durations": window_durations,
                    "mouseClicks": window_mouse_clicks,
                    "keystrokes": window_keystrokes,
                    "mic": window_mic,
                    "mouseScroll": window_mouse_scroll,
                    "camera": window_camera,
                    "app_quality": window_app_quality,
                    "employeeId": emp_id,
                    "window_start_idx": start_idx,
                }
                sequences.append(sequence)

    return sequences


def create_vocab(sequences):
    """Create vocabulary mapping from sequences."""
    unique_apps = set()
    for seq in sequences:
        unique_apps.update(seq["apps"])

    app_to_idx = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for idx, app in enumerate(sorted(unique_apps)):
        app_to_idx[app] = idx + 2

    return app_to_idx
