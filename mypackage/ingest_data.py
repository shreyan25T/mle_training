"""
Script to perform the ingestion of the data.
Takes in the script arguments from the command line.
The arguments are:
    1. path: The output path of the training and test dataset.
    2. ratio: The ratio of train and test split.
"""
import argparse
import os
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit

from custom_logger import configure_logger


def fetch_arguments():
    """
    This method handles the command line arguments entered by the user.
    It uses argparse module to parse the arguments.

    Returns:
    --------
    args(dict):
        A dictionary of command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        type=str,
        help="""Enter the path of the folder
                where the training and test
                dataset will be stored""",
        default=str(Path("../..")/"data"/"processed"),
    )
    parser.add_argument(
        "-r", "--ratio", type=float, help="""Enter the test size ratio""", default=0.2
    )
    parser.add_argument(
        "-ll",
        "--log_level",
        type=str,
        help="""Specify the log level out of the list
                ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]""",
        default="DEBUG",
    )
    parser.add_argument(
        "-tf",
        "--to_file",
        type=str,
        help="""If you want to use a file for logging, enter the file path
                """,
        default=None,
    )
    parser.add_argument(
        "-nc",
        "--no_console",
        action="store_true",
        help="""State whether logging will be done to console or not""",
    )

    args = vars(parser.parse_args())
    return args


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = str(Path("../..")/"data"/"raw")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    To download the housing tgz file and then extracting
    its content into a given folder.

    This moduel downloads the required dataset, housing.tgz using the link
    https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing

    After downloading the .tgz file, the modules extracts it.

    Args:
    -----
    housing_url (str, optional):
        URL for the housing dataset. Defaults to HOUSING_URL.
    housing_path (str, optional):
        Path where the dataset will be extracted to from the housing.tgz file.
        Defaults to HOUSING_PATH.
    """
    os.makedirs(housing_path, exist_ok=True)

    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)

    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)


def load_housing_data(housing_path=HOUSING_PATH):
    """
    To read the downloaded housing dataset.

    Args:
    -----
    housing_path (str, optional):
        Path to the housing dataset. Defaults to HOUSING_PATH.

    Returns:
    --------
    pandas.DataFrame:
        .csv file as a pandas dataframe
    """
    csv_path = os.path.join(housing_path, "housing.csv")

    return pd.read_csv(csv_path)


def create_train_test(data, output_path, test_size=0.2):
    """
    Creates a Stratified Shuffle Split of the housing dataset.
    The dataset is converted into train.csv and test.csv
    with a given test size ratio.

    The converted .csv files are stored in the output_path given as
    input by the user. Later, this path itself will be used for reading
    train.csv and test.csv

    Args:
    -----
    data (pandas.DataFrame):
        Housing dataset that needs to be split.
    output_path (str):
        Path where the train and test dataset will be created.
    test_size (float, optional):
        The ratio in which the data will be divided.
        Defaults to 0.2.
    """
    housing = data

    housing["income_cat"] = pd.cut(
        x=housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    # Creating train and test dataset
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    try:
        os.makedirs(output_path, exist_ok=True)
        strat_train_set.to_csv(os.path.join(output_path, "train.csv"), index=False)
        strat_test_set.to_csv(os.path.join(output_path, "test.csv"), index=False)

    except Exception as e:
        print(f"Could not create datasets. {e}")


def main(args, logger):
    # Creating the dataset
    fetch_housing_data()
    logger.debug("Downloading housing.tgz")
    logger.info(f"Dataset extracted to {HOUSING_PATH}")

    df = load_housing_data()
    logger.debug("housing.csv returned")

    # Storing training and test dataset
    create_train_test(df, args["path"], args["ratio"])
    logger.debug(f"train.csv and test.csv created in {args['path']} succesfully!")


if __name__ == "__main__":
    args = fetch_arguments()

    my_logger = configure_logger(
        args["log_level"], args["to_file"], not args["no_console"]
    )

    main(args, my_logger)
