import argparse
from asyncio.log import logger
import logging
import os
from pathlib import Path

import joblib as jb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from custom_logger import configure_logger
from processing import prepare_data


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
        "-mp",
        "--mpath",
        type=str,
        help=""""Enter the path where the
                 dumped models are stored.""",
        default=str(Path("../..")/"models"),
    )
    parser.add_argument(
        "-dp",
        "--data",
        type=str,
        help=""""Enter the path where the
                 testing dataset is stored.""",
        default=str(Path("../..")/"data"/"processed"),
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


def load_test_data(path):
    """
    To load the test dataset for scoring the trained models.

    Args:
    ------
        path (str):
            Path to the directory in which test dataset is stored.

    Returns:
    --------
        pandas.DataFrame: loaded test dataset
    """
    test_path = os.path.join(path, "test.csv")
    test = pd.read_csv(test_path)
    return test


def read_models(path):
    """
    Reading the dumped models to score them using the test dataset.


    Args:
    -----
    path (str): Path to the dumped models

    Returns:
    --------

    tuple: A tuple of all the models.
    """
    model_paths = (os.path.join(path, model) for model in os.listdir(path))
    models = (jb.load(model_path) for model_path in model_paths)
    return models


def get_predictions(models, X, y):
    """
    To score all the models that were loaded using joblib.
    This method prints all the metrics and the accuracy of all the models.

    Args:
    -----
    models (tuple): A tuple of models.
    X (numpy.array): Independent variables
    y (numpy.array): Dependent variables


    """
    for m in models:
        y_pred = m.predict(X)

        model_score = m.score(y_pred, y)

        logger.info(f"For {m}, accuaracy: {model_score}")
        mae = mean_absolute_error(y_pred, y)
        logger.info(f"MAE: {mae}")

        rmse = np.sqrt(mean_squared_error(y_pred, y))
        logger.info(f"RMSE: {rmse}")


def main(args, logger):
    # Reading the data
    test = load_test_data(args["data"])
    logger.debug("Test data loaded for scoring!")

    # Preparing the data
    X_test, y_test = prepare_data(test)
    logger.debug("X_test and y_test prepared")

    # Reading the models
    models = read_models(args["mpath"])
    logger.debug(f"{len(list(models))} model(s) loaded successfully!")

    # Give predictions
    get_predictions(models, X_test, y_test)
    logger.info("predictions and various metrics got evaluated.")


if __name__ == "__main__":
    args = fetch_arguments()
    my_logger = configure_logger(
        args["log_level"], args["to_file"], not args["no_console"]
    )

    main(args, my_logger)


