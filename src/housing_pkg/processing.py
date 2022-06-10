"""
Contains all the methods like various custom transformers and pipelines
to prepare the data for  training and testing the model.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    To add three new attributes using the current
    available attributes/features/columns in the original dataframe.

    Args:
    -----
    BaseEstimator:
        Base class for all estimators in scikit-learn
    TransformerMixin:
        Mixin class for all transformers in scikit-learn.
    """

    def __init__(self, add_bedrooms_per_room=True):
        """
        To initialize the CombinedAttryibutesAdder object.

        Args:
        -----
        add_bedrooms_per_room(bool, optional):
            State whether `bedrooms_per_room` attribute will be added or not.
            Defaults to True.
        """
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        """
        To fit the custom Transformer on the given data.
        Also to calculate the indices of top k elements.

        Args:
        -----
        X (pd.DataFrame):
            Dataframe of independent variables in a dataset.
        y (pd.DataFrame, optional):
            Dataframe of depenedent variables in the same dataset as above.
            Defaults to None.

        """
        return self

    def transform(self, X, y=None):
        """
        To transform the fitted data into a new numpy array
        containing extra attributes.
        These attributes are:
        1. rooms_per_household
        2. population_per_household
        3. bedrooms_per_rooms

        Args:
        -----
        X (pd.DataFrame):
            Dataframe of independent variables in a dataset.
        y (pd.DataFrame, optional):
            Dataframe of depenedent variables in the same dataset as above.
            Defaults to None.

        Returns:
        --------
        numpy.array:
            An array containing extra added attributes.
        """
        room_per_hhold = X[:, rooms_ix] / X[:, households_ix]
        pop_per_household = X[:, population_ix] / X[:, households_ix]

        if self.add_bedrooms_per_room:
            bed_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, room_per_hhold, pop_per_household, bed_per_room]
        else:
            return np.c_[X, room_per_hhold, pop_per_household]


def prepare_data(data):
    """
    To prepare the data using scikit-learn pipelines.
    This method performs the following transformations
    1. Adds some extra attributes, which are just the
       mathematical combinations of the existing attributes
       in the data, to the end of the data.
    2. Imputes the missing values by taking the median
       of the values in each column.
    3. Uses standard scaler to scale the numeric values.
    4. Performs one hot encoding on the categorical variables/columns.
    5. Splits the data into dependent(y) and independent(X) variables.

    Args:
    -----
    data (pandas.DataFrame):
        Data that needs to be prepared.

    Returns:
    --------
    X_prepared(numpy.array):
        The transformed data including extra attributes.
    y: The target variable of the data.
    """
    X = data.drop("median_house_value", axis=1)
    y = data["median_house_value"].copy()

    X_num = X.drop("ocean_proximity", axis=1)

    # Processing numeric data
    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", CombinedAttributesAdder()),
            ("std_scaler", StandardScaler()),
        ]
    )

    num_attribs = list(X_num)
    cat_attribs = ["ocean_proximity"]

    # Processing the columns with specific data type (num, cat)
    full_pipeline = ColumnTransformer(
        [
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs),
        ]
    )

    # Transforming the data
    X_prepared = full_pipeline.fit_transform(X)
    return X_prepared, y
