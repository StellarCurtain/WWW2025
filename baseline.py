# Imports and library setup.
import os
from datetime import datetime, timezone, timedelta
from multiprocessing import Pool
from typing import Union, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from tqdm import tqdm

SN_TYPE = "B"  # Device type (either "A" or "B"); using type A as an example
TEST_PHASE = 1  # Test phase (either 1 or 2); using phase 1 as an example

# Setup dataset and working directories, and display input data information.
DATASET_DIR = "dataset"
WORK_DIR = f"yyt/save{SN_TYPE}"

# Create necessary directories if they don't exist
os.makedirs(os.path.join(WORK_DIR, "positive_train_files"), exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, "negative_train_files"), exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, "test_files"), exist_ok=True)
os.makedirs(os.path.join(WORK_DIR, "processed_data"), exist_ok=True)

print("Dataset directory listing:", os.listdir(DATASET_DIR))
print("Type A file count:", len(os.listdir(os.path.join(DATASET_DIR, "type_A"))))
print("Type B file count:", len(os.listdir(os.path.join(DATASET_DIR, "type_B"))))

# Define time constants, date ranges, and read failure ticket information.
ONE_MINUTE = 60  # Seconds in one minute
ONE_HOUR = 3600  # Seconds in one hour
ONE_DAY = 86400  # Seconds in one day

# SN_TYPE = "A"  # Device type (either "A" or "B"); using type A as an example
# TEST_PHASE = 1  # Test phase (either 1 or 2); using phase 1 as an example

TRAIN_DATE_RANGE: Tuple[str, str] = ("2024-01-01", "2024-06-01")
if TEST_PHASE == 1:
    TEST_DATE_RANGE: Tuple[str, str] = (
        "2024-06-01",
        "2024-08-01",
    )  # Test data range for phase 1
else:
    TEST_DATE_RANGE: Tuple[str, str] = (
        "2024-08-01",
        "2024-10-01",
    )  # Test data range for phase 2

# Read failure tickets CSV and create a mapping from sensor ID to alarm time.
failure_tickets = pd.read_csv(os.path.join(DATASET_DIR, "failure_ticket.csv"))
sn_alarm_map = {
    sn: alarm_time
    for sn, alarm_time in zip(failure_tickets["sn_name"], failure_tickets["alarm_time"])
}

# Define a function to convert a date string to a Unix timestamp using UTC+8 timezone.
def convert_date_to_timestamp(date_str: str) -> int:
    """
    Convert a date string in the format 'YYYY-MM-DD' to a Unix timestamp using UTC+8 timezone.

    :param date_str: Date string in 'YYYY-MM-DD' format.
    :return: Unix timestamp as an integer.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    dt = dt.replace(tzinfo=timezone(timedelta(hours=8)))  # Set timezone to UTC+8
    return int(dt.timestamp())


# Calculate timestamp boundaries for training and testing data.
TRAIN_START_TS = convert_date_to_timestamp(TRAIN_DATE_RANGE[0])
TRAIN_END_TS = convert_date_to_timestamp(TRAIN_DATE_RANGE[1])
TEST_START_TS = convert_date_to_timestamp(TEST_DATE_RANGE[0])
TEST_END_TS = convert_date_to_timestamp(TEST_DATE_RANGE[1])

# Define a function to split log data into positive training, negative training, and test datasets.
def split_logs_by_time_window(
    log_df: pd.DataFrame, serial_number: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split log data into positive training, negative training, and test sets based on time windows.

    - Positive training data: logs between (alarm_time - 30 days) and alarm_time (if an alarm time exists).
    - Negative training data: logs between (TRAIN_END_TS - 60 days) and (TRAIN_END_TS - 30 days) if no alarm time exists.
    - Test data: logs between TEST_START_TS and TEST_END_TS.

    :param log_df: DataFrame containing log entries with a 'LogTime' column (Unix timestamps).
    :param serial_number: Sensor identifier to lookup in the alarm map.
    :return: Tuple of (positive_df, negative_df, test_df)
    """
    positive_df = pd.DataFrame()
    negative_df = pd.DataFrame()
    test_df = pd.DataFrame()

    train_df = log_df[
        (log_df["LogTime"] >= TRAIN_START_TS) & (log_df["LogTime"] < TRAIN_END_TS)
    ]
    test_df = log_df[
        (log_df["LogTime"] >= TEST_START_TS) & (log_df["LogTime"] < TEST_END_TS)
    ]

    if not train_df.empty:
        if serial_number in sn_alarm_map:
            alarm_time = sn_alarm_map[serial_number]
            positive_df = train_df[
                (train_df["LogTime"] >= alarm_time - 30 * ONE_DAY)
                & (train_df["LogTime"] < alarm_time)
            ]
        else:
            negative_df = train_df[
                (train_df["LogTime"] >= TRAIN_END_TS - 60 * ONE_DAY)
                & (train_df["LogTime"] < TRAIN_END_TS - 30 * ONE_DAY)
            ]

    return positive_df, negative_df, test_df

# Define a function to process a single log file by splitting it and saving the resulting datasets.
def process_log_file(args):
    filename, dataset_dir, SN_TYPE, working_dir = args
    file_path = os.path.join(dataset_dir, f"type_{SN_TYPE}", filename)
    log_df = pd.read_feather(file_path)

    serial_number = os.path.splitext(filename)[0]
    index_list = [(serial_number, log_time) for log_time in log_df["LogTime"]]
    log_df.index = pd.MultiIndex.from_tuples(index_list)
    pos_df, neg_df, test_df = split_logs_by_time_window(log_df, serial_number)

    if not pos_df.empty:
        output_path = os.path.join(
            working_dir, "positive_train_files", f"{serial_number}.feather"
        )
        pos_df.to_feather(output_path)
    if not neg_df.empty:
        output_path = os.path.join(
            working_dir, "negative_train_files", f"{serial_number}.feather"
        )
        neg_df.to_feather(output_path)
    if not test_df.empty:
        output_path = os.path.join(working_dir, "test_files", f"{serial_number}.feather")
        test_df.to_feather(output_path)

    del pos_df, neg_df, test_df

# Process all log files in parallel using multiprocessing.
# Important: only 5,000 SNs are selected in the Notebook. Please consider modifying it to include all SNs.
log_files = os.listdir(os.path.join(DATASET_DIR, f"type_{SN_TYPE}"))
process_args = [(file, DATASET_DIR, SN_TYPE, WORK_DIR) for file in log_files]

print("Starting processing of log files...")
with Pool(processes=4) as pool:
    list(tqdm(pool.imap_unordered(process_log_file, process_args), total=len(log_files)))

# Define functions for parallel concatenation of DataFrame chunks.
CHUNK_SIZE = 200


def concat_chunks(chunks: List[pd.DataFrame]) -> Union[pd.DataFrame, None]:
    """
    Concatenate a list of DataFrame chunks.


    :param chunks: List of DataFrames.
    :return: Concatenated DataFrame, or None if the list is empty.
    """
    valid_chunks = [chunk for chunk in chunks if chunk is not None]
    if valid_chunks:
        return pd.concat(valid_chunks)
    return None


def parallel_concatenate(
    dataframes: List, chunk_size: int = CHUNK_SIZE
) -> Union[pd.DataFrame, None]:
    """
    Parallel concatenation of a list of DataFrames.

    :param dataframes: List of DataFrames to concatenate.
    :param chunk_size: Number of DataFrames per chunk.
    :return: Concatenated DataFrame.
    """
    chunks = [
        dataframes[i : i + chunk_size] for i in range(0, len(dataframes), chunk_size)
    ]
    with Pool() as pool:
        concatenated_chunks = pool.map(concat_chunks, chunks)
    return concat_chunks(concatenated_chunks)

# Define functions to load feather files and process them in parallel.
def load_feather_file(args):
    filename, directory_path = args
    file_path = os.path.join(directory_path, filename)
    df = pd.read_feather(file_path)
    return df if not df.empty else None


def load_and_concat_files(file_list, directory_path):
    args_list = [(filename, directory_path) for filename in file_list]
    with Pool(processes=4) as pool:
        dataframes = list(
            tqdm(
                pool.imap_unordered(load_feather_file, args_list),
                total=len(file_list),
                desc="Loading feather files",
            )
        )
    return parallel_concatenate(dataframes)

# Define a function to process positive training data.
def process_positive_training_data(working_dir: str) -> None:
    """
    Load, label, and save the positive training data.
    """
    positive_train_dir = os.path.join(working_dir, "positive_train_files")
    processed_data_dir = os.path.join(working_dir, "processed_data")
    positive_file_list = os.listdir(positive_train_dir)

    print("Processing positive training data...")
    pos_df = load_and_concat_files(positive_file_list, positive_train_dir)
    if pos_df is not None:
        pos_df["label"] = 1
        output_path = os.path.join(processed_data_dir, "positive_train.feather")
        pos_df.to_feather(output_path)
        print(f"Saved positive training data to {output_path}.")
    else:
        print("No positive training data found.")

    del pos_df

process_positive_training_data(WORK_DIR)

# Define a function to process negative training data.
def process_negative_training_data(working_dir: str) -> None:
    """
    Load, label, and save the negative training data.
    """
    negative_train_dir = os.path.join(working_dir, "negative_train_files")
    processed_data_dir = os.path.join(working_dir, "processed_data")
    negative_file_list = os.listdir(negative_train_dir)

    print("Processing negative training data...")
    neg_df = load_and_concat_files(negative_file_list, negative_train_dir)
    if neg_df is not None:
        neg_df["label"] = 0
        output_path = os.path.join(processed_data_dir, "negative_train.feather")
        neg_df.to_feather(output_path)
        print(f"Saved negative training data to {output_path}.")
    else:
        print("No negative training data found.")
        
    del neg_df

process_negative_training_data(WORK_DIR)

# Define a function to process test data in chunks.
def process_test_data(working_dir: str, num_chunks: int = 8) -> None:
    """
    Load and save test data in chunks.

    :param working_dir: The base working directory.
    :param num_chunks: The number of chunks to split the test files into.
    """
    test_dir = os.path.join(working_dir, "test_files")
    processed_data_dir = os.path.join(working_dir, "processed_data")
    test_file_list = os.listdir(test_dir)
    split_test_files = np.array_split(test_file_list, num_chunks)

    for chunk_index, file_chunk in enumerate(split_test_files):
        print(f"Processing test files chunk {chunk_index + 1} of {num_chunks}...")
        test_chunk_df = load_and_concat_files(file_chunk, test_dir)
        if test_chunk_df is not None:
            output_path = os.path.join(processed_data_dir, f"test_{chunk_index}.feather")
            test_chunk_df.to_feather(output_path)
            print(f"Saved test data chunk {chunk_index} to {output_path}.")
        else:
            print(f"No data found in test files chunk {chunk_index}.")
        del test_chunk_df

process_test_data(WORK_DIR, num_chunks=8)

# Initialize the LightGBM model with specified parameters.
lgbm_params = {
    "learning_rate": 0.02,
    "n_estimators": 500,
    "max_depth": 8,
    "num_leaves": 20,
    "min_child_samples": 20,
    "verbose": 1,
}
print("Initializing LightGBM model with parameters:", lgbm_params)
model = LGBMClassifier(**lgbm_params)

train_pos = pd.read_feather(
    os.path.join(WORK_DIR, "processed_data", "positive_train.feather")
)
train_neg = pd.read_feather(
    os.path.join(WORK_DIR, "processed_data", "negative_train.feather")
)

train_all = pd.concat([train_pos, train_neg])
train_all.drop("LogTime", axis=1, inplace=True)
train_all = train_all.sort_index(axis=1)

model.fit(train_all.drop(columns=["label"]), train_all["label"])
del train_all

result = {}
THRESHOLD = 0.5

test_data_path = os.path.join(WORK_DIR, "processed_data")
for file in os.listdir(test_data_path):
    test_df = pd.read_feather(os.path.join(test_data_path, file))
    test_df["sn_name"] = test_df.index.get_level_values(0)
    test_df["log_time"] = test_df.index.get_level_values(1)

    test_df = test_df[model.feature_name_]
    predict_result = model.predict_proba(test_df)

    index_list = list(test_df.index)
    for i in tqdm(range(len(index_list))):
        p_s = predict_result[i][1]

        # Filter low-probability samples to reduce memory usage of predictions
        if p_s < 0.1:
            continue

        sn = index_list[i][0]
        sn_t = datetime.fromtimestamp(index_list[i][1])
        result.setdefault(sn, [])
        result[sn].append((sn_t, p_s))

# Filter the prediction results based on the threshold
result = {
    sn: [int(sn_t.timestamp()) for sn_t, p_s in pred_list if p_s >= THRESHOLD]
    for sn, pred_list in result.items()
}

# Filter out empty prediction results and sort the results by time
result = {
    sn: sorted(pred_list) for sn, pred_list in result.items() if pred_list
}

# Convert prediction results into submission format
submission = []
for sn in result:  # Iterate through prediction results for each SN
    for timestamp in result[sn]:  # Iterate through each timestamp
        submission.append([sn, timestamp, SN_TYPE])  # Add SN name, prediction timestamp, and SN type

# Convert submission data into a DataFrame and save it as a CSV file
submission = pd.DataFrame(
    submission, columns=["sn_name", "prediction_timestamp", "serial_number_type"]
)
submission.to_csv(os.path.join(WORK_DIR, "submissionA.csv"), index=False, encoding="utf-8")

#import zipfile

#csv_file = os.path.join(WORK_DIR, "submission.csv")
#zip_file = os.path.join(WORK_DIR, "submission.zip")

#with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
#    zipf.write(csv_file, os.path.basename(csv_file))

#print(f"{csv_file} has been compressed into {zip_file}.")