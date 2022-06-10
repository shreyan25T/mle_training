import argparse
import os
from pathlib import Path

import joblib as jb
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

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
    # Taking the arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ip",
        "--ipath",
        type=str,
        help="""Enter the path of the folder
                where the train and test
                dataset are stored""",
        default=str(Path("../..")/"data"/"processed"),
    )
    parser.add_argument(
        "-op",
        "--opath",
        type=str,
        help="""Enter the path of the folder
                where the model pickles will be
                stored""",
        default=str(Path("../..")/"models"),
    )
    parser.add_argument(
        "-m",
        "--model",
        default="lin_reg",
        type=str,
        help="""Choose the regressor that you want to train from the list
                [lin_reg, tree_reg, forest_reg]""",
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


def train_model(estimator, X=None, y=None):
    """
    Fits a given model with the given data.
    The method contains three scikit-learn models.
    1. Linear Regressor
    2. DecisionTree Regressor
    3. RandomForest Regressor

    For RandomForest Regressor only, hyperparamter tuning takes place.

    Hyperparameter tuning is done using scikit-learn's RandomizedSearchCV
    which accepts a distribution of differenet parameters for a Random
    Forest Regressor.

    Args:
    -----
    estimator (str):
        The name of the estimator out of the list
        [lin_reg, forest_reg, tree_reg]
    X (numpy.array, optional):
        An array of independent variables. Defaults to None.
    y (numpy.array, optional):
        An array of the target variables. Defaults to None.

    Returns:
    --------
        model (sklearn.model):
            Returns a trained model.
    """
    models = {
        "lin_reg": LinearRegression(),
        "tree_reg": DecisionTreeRegressor(random_state=42),
        "forest_reg": RandomForestRegressor(random_state=42),
    }

    if estimator == "forest_reg":
        param_distribs = {
            "n_estimators": randint(low=1, high=200),
            "max_features": randint(low=1, high=8),
        }

        rnd_search = RandomizedSearchCV(
            models[estimator],
            param_distributions=param_distribs,
            n_iter=10,
            cv=5,
            scoring="neg_mean_squared_error",
            verbose=2,
        )

        rnd_search.fit(X, y)
        best_model = rnd_search.best_estimator_
        return best_model
    else:
        models[estimator].fit(X, y)
        return models[estimator]


def dump_model(model, path):
    """
    To dump the trained model as a file, which can be used to give predictions.
    Dumping takes place using the joblib package.

    Args:
    -----

    model (sklearn.model): Model to dump
    path (str): Where the dumped file will be saved.
    """
    try:
        os.makedirs(path, exist_ok=True)
        jb.dump(model, os.path.join(path, f"model_{model}.pkl"))
    except Exception as e:
        print(e)


def load_data(path):
    """
    Loading the test and train dataset from the given path.

    Args:
    -----

    path (str): Where the train.csv amd test.csv are located.

    Returns:
    --------

    tuple: containing pandas.DataFrame, one file for each of
           train.csv and test.csv
    """
    train_data = pd.read_csv(os.path.join(path, "train.csv"))
    test_data = pd.read_csv(os.path.join(path, "test.csv"))

    return train_data, test_data


def main(args, logger):
    # Reading the data using load_data()
    housing_train, housing_test = load_data(args["ipath"])
    logger.debug("train.csv and test.csv loaded.")

    # Perparing the data using prepare_data()
    X_train, y_train = prepare_data(housing_train)
    logger.debug("X_train and y_train data prepared.")

    # Training the model using train_model()
    trained_model = train_model(args["model"], X=X_train, y=y_train)
    logger.warning(f"training of {args['model']} complete. ")

    # Dumping the model using dump_model()
    dump_model(trained_model, args["opath"])
    logger.info(f"model successfully dumped at {args['opath']}")


if __name__ == "__main__":
    args = fetch_arguments()

    my_logger = configure_logger(
        args["log_level"], args["to_file"], not args["no_console"]
    )

    main(args, my_logger)
