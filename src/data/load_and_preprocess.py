import pandas as pd
from pandas import Timestamp
import numpy as np
from tqdm import tqdm


def load_raw_data(transactions_file, app_mappings_file):
    """Load raw data from CSV files."""
    df_day_point = pd.read_csv(transactions_file, sep=";", index_col=0)
    df_app_to_class = pd.read_csv(app_mappings_file)
    return df_day_point, df_app_to_class


def process_row(row):
    """Process a single row of the dataframe to create exploded format."""
    apps = eval(row["app"])
    durations = eval(row["app_durations"])
    start_times = eval(row["app_start_times"])
    end_times = eval(row["app_end_times"])

    return pd.DataFrame(
        {
            "app": apps,
            "duration": durations,
            "app_start_time": start_times,
            "app_end_time": end_times,
            "employeeId": [row["employeeId"]] * len(apps),
            "workday_start": [row["start_time"]] * len(apps),
            "workday_end": [row["end_time"]] * len(apps),
            "workday_duration": [row["workday_duration"]] * len(apps),
            "hours_until_next_workday": [row["hours_until_next_workday"]] * len(apps),
        }
    )


def create_exploded_df(df_day_point):
    """Create exploded dataframe from day point data."""
    exploded_df = pd.concat(
        [process_row(row) for _, row in df_day_point.iterrows()], ignore_index=True
    )
    exploded_df["duration"] = exploded_df["duration"] * 60  # Convert to seconds
    return exploded_df


def create_sequences(exploded_df, window_size=64, stride=8):
    """Create sequences using sliding window approach."""
    df_sorted = exploded_df.sort_values(["employeeId", "app_start_time"])
    sequences = []

    for (emp_id,), group in tqdm(
        df_sorted.groupby(["employeeId"]), desc="Creating sequences"
    ):
        workday_start_time = pd.to_datetime(group["workday_start"].iloc[0])
        delta_from_workday_start = np.log2(
            np.maximum(
                1,
                (
                    (
                        pd.to_datetime(group["app_start_time"]) - workday_start_time
                    ).dt.total_seconds()
                ),
            )
        ).tolist()

        apps = group["app"].tolist()
        # durations = np.log2(np.maximum(1, (pd.to_datetime(group['app_end_time']) -
        #                                  pd.to_datetime(group['app_start_time'])).dt.total_seconds())).tolist()
        durations = (
            (
                pd.to_datetime(group["app_end_time"])
                - pd.to_datetime(group["app_start_time"])
            ).dt.total_seconds()
        ).tolist()

        for start_idx in range(0, len(apps), stride):
            end_idx = start_idx + window_size
            window_apps = apps[start_idx:end_idx]
            window_durations = durations[start_idx:end_idx]

            if len(window_apps) >= 4:
                sequence = {
                    "apps": window_apps,
                    "durations": window_durations,
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
