import os
import tarfile
import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
hs_PATH = os.path.join("datasets", "hs")
hs_URL = DOWNLOAD_ROOT + "datasets/hs/hs.tgz"


def fetch_hs_data(hs_url=hs_URL, hs_path=hs_PATH):
    os.makedirs(hs_path, exist_ok=True)
    tgz_path = os.path.join(hs_path, "hs.tgz")
    urllib.request.urlretrieve(hs_url, tgz_path)
    hs_tgz = tarfile.open(tgz_path)
    hs_tgz.extractall(path=hs_path)
    hs_tgz.close()


def load_hs_data(hs_path=hs_PATH):
    csv_path = os.path.join(hs_path, "hs.csv")
    return pd.read_csv(csv_path)


hs = load_hs_data


train_set, test_set = train_test_split(hs, test_size=0.2, random_state=42)

hs["income_cat"] = pd.cut(
    hs["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(hs, hs["income_cat"]):
    strat_train_set = hs.loc[train_index]
    strat_test_set = hs.loc[test_index]


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


train_set, test_set = train_test_split(hs, test_size=0.2, random_state=42)

compare_props = pd.DataFrame(
    {
        "Overall": income_cat_proportions(hs),
        "Stratified": income_cat_proportions(strat_test_set),
        "Random": income_cat_proportions(test_set),
    }
).sort_index()
compare_props["Rand. %error"] = (
    100 * compare_props["Random"] / compare_props["Overall"] - 100
)
compare_props["Strat. %error"] = (
    100 * compare_props["Stratified"] / compare_props["Overall"] - 100
)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

hs = strat_train_set.copy()
hs.plot(kind="scatter", x="longitude", y="latitude")
hs.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

corr_matrix = hs.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
hs["rooms_per_household"] = hs["total_rooms"] / hs["households"]
hs["bedrooms_per_room"] = hs["total_bedrooms"] / hs["total_rooms"]
hs["population_per_household"] = hs["population"] / hs["households"]

hs = strat_train_set.drop(
    "median_house_value", axis=1
)  # drop labels for training set
hs_labels = strat_train_set["median_house_value"].copy()

imputer = SimpleImputer(strategy="median")

hs_num = hs.drop("ocean_proximity", axis=1)

imputer.fit(hs_num)
X = imputer.transform(hs_num)

hs_tr = pd.DataFrame(X, columns=hs_num.columns, index=hs.index)
hs_tr["rooms_per_household"] = hs_tr["total_rooms"] / hs_tr["households"]
hs_tr["bedrooms_per_room"] = (
    hs_tr["total_bedrooms"] / hs_tr["total_rooms"]
)
hs_tr["population_per_household"] = (
    hs_tr["population"] / hs_tr["households"]
)

hs_cat = hs[["ocean_proximity"]]
hs_prepared = hs_tr.join(pd.get_dummies(hs_cat, drop_first=True))

lin_reg = LinearRegression()
lin_reg.fit(hs_prepared, hs_labels)

hs_predictions = lin_reg.predict(hs_prepared)
lin_mse = mean_squared_error(hs_labels, hs_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

lin_mae = mean_absolute_error(hs_labels, hs_predictions)
lin_mae

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(hs_prepared, hs_labels)

hs_predictions = tree_reg.predict(hs_prepared)
tree_mse = mean_squared_error(hs_labels, hs_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

param_distribs = {
    "n_estimators": randint(low=1, high=200),
    "max_features": randint(low=1, high=8),
}

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(
    forest_reg,
    param_distributions=param_distribs,
    n_iter=10,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
)
rnd_search.fit(hs_prepared, hs_labels)
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(
    forest_reg,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    return_train_score=True,
)
grid_search.fit(hs_prepared, hs_labels)

grid_search.best_params_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
sorted(zip(feature_importances, hs_prepared.columns), reverse=True)

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_num = X_test.drop("ocean_proximity", axis=1)
X_test_prep = imputer.transform(X_test_num)
X_test_prep = pd.DataFrame(
    X_test_prep, columns=X_test_num.columns, index=X_test.index
)
X_test_prep["rooms_per_household"] = (
    X_test_prep["total_rooms"] / X_test_prep["households"]
)
X_test_prep["bedrooms_per_room"] = (
    X_test_prep["total_bedrooms"] / X_test_prep["total_rooms"]
)
X_test_prep["population_per_household"] = (
    X_test_prep["population"] / X_test_prep["households"]
)

X_test_cat = X_test[["ocean_proximity"]]
X_test_prep = X_test_prep.join(pd.get_dummies(X_test_cat, drop_first=True))


final_predictions = final_model.predict(X_test_prep)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
