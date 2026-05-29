# %% [markdown]
# # TAM400 - Predicting Vehicle Fuel Efficiency
#
# This file is written in the same step-by-step style as the course labs.
# It trains a regression model on the UCI Auto MPG dataset and saves the
# final trained model so the FastAPI app can use it.
#
# Main idea:
# - Input: vehicle attributes such as weight, horsepower, cylinders, origin.
# - Output: predicted fuel efficiency in MPG.

# %%
import json
import sys
from pathlib import Path

assert sys.version_info >= (3, 7)

from packaging import version
import sklearn

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

matplotlib.use("Agg")

plt.rc("font", size=14)
plt.rc("axes", labelsize=14, titlesize=14)
plt.rc("legend", fontsize=14)
plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)

# %% [markdown]
# ## Task 1: Create folders and helper function for saving figures

# %%
FIGURES_PATH = Path("artifacts") / "figures" / "lab_style"
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

MODELS_PATH = Path("artifacts") / "models"
MODELS_PATH.mkdir(parents=True, exist_ok=True)

METRICS_PATH = Path("artifacts") / "metrics"
METRICS_PATH.mkdir(parents=True, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = FIGURES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def rmse_score(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# %% [markdown]
# ## Task 2: Load the Auto MPG dataset

# %%
from pathlib import Path
from urllib.request import urlopen
import ssl
import pandas as pd
import numpy as np

AUTO_MPG_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

RAW_DATA_PATH = Path("data") / "raw" / "auto-mpg.data"
RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

columns = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
    "car_name",
]

if not RAW_DATA_PATH.exists():
    print("Downloading Auto MPG dataset...")

    try:
        with urlopen(AUTO_MPG_URL) as response:
            RAW_DATA_PATH.write_bytes(response.read())

    except Exception as error:
        print("Normal download failed. Trying without SSL verification...")
        context = ssl._create_unverified_context()
        https_url = AUTO_MPG_URL.replace("http://", "https://")

        with urlopen(https_url, context=context) as response:
            RAW_DATA_PATH.write_bytes(response.read())

rows = []

with RAW_DATA_PATH.open("r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split(maxsplit=8)
        rows.append([*parts[:8], parts[8].strip('"')])

auto_mpg = pd.DataFrame(rows, columns=columns)

numeric_columns = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
]

for column in numeric_columns:
    auto_mpg[column] = pd.to_numeric(
        auto_mpg[column].replace("?", np.nan),
        errors="coerce"
    )

print("Dataset loaded successfully!")
print(auto_mpg.head())

# %%
print("Dataset shape:", auto_mpg.shape)
print("\nMissing values:")
print(auto_mpg.isna().sum())

#%% [markdown]
# ## Task 3: Quick exploratory data analysis

# %%
auto_mpg["mpg"].hist(bins=20)
plt.xlabel("MPG")
plt.ylabel("Number of cars")
plt.title("Distribution of fuel efficiency")
save_fig("mpg_distribution")
plt.close()

# %%
plt.scatter(auto_mpg["weight"], auto_mpg["mpg"], alpha=0.7)
plt.xlabel("Weight")
plt.ylabel("MPG")
plt.title("MPG decreases as vehicle weight increases")
save_fig("mpg_vs_weight")
plt.close()

# %%
plt.scatter(auto_mpg["horsepower"], auto_mpg["mpg"], alpha=0.7)
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("MPG compared to horsepower")
save_fig("mpg_vs_horsepower")
plt.close()

# %% [markdown]
# ## Task 4: Prepare features and target
#
# Target variable:
# - `mpg`
#
# Features used by the model:
# - `cylinders`
# - `displacement`
# - `horsepower`
# - `weight`
# - `acceleration`
# - `model_year`
# - `origin`
#
# We do not use `car_name` in the first model because it is mostly an ID/text field.

# %%
feature_columns = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
]

X = auto_mpg[feature_columns]
y = auto_mpg["mpg"]

print(X.head())
print(y.head())

# %% [markdown]
# ## Task 5: Split the data into training and test sets
#
# We split before preprocessing to avoid data leakage.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("Training set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)

# %% [markdown]
# ## Task 6: Build preprocessing
#
# `horsepower` has missing values, so we use `SimpleImputer`.
# `origin` is categorical, so we use `OneHotEncoder`.
# Linear regression also gets `StandardScaler` for numeric columns.

# %%
num_attribs = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
]

cat_attribs = ["origin"]


def make_preprocessor(scale_numeric=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipeline, num_attribs),
            ("cat", categorical_pipeline, cat_attribs),
        ]
    )

# %% [markdown]
# ## Task 7: Train a baseline model
#
# `DummyRegressor` is the simplest baseline. It usually predicts the mean MPG.
# A real model should perform clearly better than this.

# %%
dummy_model = Pipeline(
    [
        ("preprocess", make_preprocessor()),
        ("model", DummyRegressor(strategy="mean")),
    ]
)

dummy_model.fit(X_train, y_train)
dummy_predictions = dummy_model.predict(X_test)

print("Dummy MAE:", mean_absolute_error(y_test, dummy_predictions))
print("Dummy RMSE:", rmse_score(y_test, dummy_predictions))
print("Dummy R2:", r2_score(y_test, dummy_predictions))

# %% [markdown]
# ## Task 8: Train Linear Regression

# %%
linear_model = Pipeline(
    [
        ("preprocess", make_preprocessor(scale_numeric=True)),
        ("model", LinearRegression()),
    ]
)

linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

print("Linear Regression MAE:", mean_absolute_error(y_test, linear_predictions))
print("Linear Regression RMSE:", rmse_score(y_test, linear_predictions))
print("Linear Regression R2:", r2_score(y_test, linear_predictions))

# %%
linear_rmse_scores = -cross_val_score(
    linear_model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_root_mean_squared_error",
)

print("Linear Regression CV RMSE scores:", linear_rmse_scores)
print("Linear Regression CV RMSE mean:", linear_rmse_scores.mean())

# %% [markdown]
# ## Task 9: Train Decision Tree Regressor

# %%
tree_model = Pipeline(
    [
        ("preprocess", make_preprocessor()),
        ("model", DecisionTreeRegressor(max_depth=5, min_samples_leaf=2, random_state=42)),
    ]
)

tree_model.fit(X_train, y_train)
tree_predictions = tree_model.predict(X_test)

print("Decision Tree MAE:", mean_absolute_error(y_test, tree_predictions))
print("Decision Tree RMSE:", rmse_score(y_test, tree_predictions))
print("Decision Tree R2:", r2_score(y_test, tree_predictions))

# %%
tree_rmse_scores = -cross_val_score(
    tree_model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_root_mean_squared_error",
)

print("Decision Tree CV RMSE scores:", tree_rmse_scores)
print("Decision Tree CV RMSE mean:", tree_rmse_scores.mean())

# %% [markdown]
# ## Task 10: Train Random Forest Regressor
#
# Random Forest is an ensemble of many decision trees. It is usually more stable
# than a single decision tree.

# %%
forest_model = Pipeline(
    [
        ("preprocess", make_preprocessor()),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                n_jobs=1,
            ),
        ),
    ]
)

forest_model.fit(X_train, y_train)
forest_predictions = forest_model.predict(X_test)

print("Random Forest MAE:", mean_absolute_error(y_test, forest_predictions))
print("Random Forest RMSE:", rmse_score(y_test, forest_predictions))
print("Random Forest R2:", r2_score(y_test, forest_predictions))

# %%
forest_rmse_scores = -cross_val_score(
    forest_model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_root_mean_squared_error",
)

print("Random Forest CV RMSE scores:", forest_rmse_scores)
print("Random Forest CV RMSE mean:", forest_rmse_scores.mean())

# %% [markdown]
# ## Task 11: Compare all models

# %%
model_results = pd.DataFrame(
    [
        {
            "model": "DummyRegressor",
            "test_mae": mean_absolute_error(y_test, dummy_predictions),
            "test_rmse": rmse_score(y_test, dummy_predictions),
            "test_r2": r2_score(y_test, dummy_predictions),
        },
        {
            "model": "LinearRegression",
            "test_mae": mean_absolute_error(y_test, linear_predictions),
            "test_rmse": rmse_score(y_test, linear_predictions),
            "test_r2": r2_score(y_test, linear_predictions),
            "cv_rmse_mean": linear_rmse_scores.mean(),
        },
        {
            "model": "DecisionTreeRegressor",
            "test_mae": mean_absolute_error(y_test, tree_predictions),
            "test_rmse": rmse_score(y_test, tree_predictions),
            "test_r2": r2_score(y_test, tree_predictions),
            "cv_rmse_mean": tree_rmse_scores.mean(),
        },
        {
            "model": "RandomForestRegressor",
            "test_mae": mean_absolute_error(y_test, forest_predictions),
            "test_rmse": rmse_score(y_test, forest_predictions),
            "test_r2": r2_score(y_test, forest_predictions),
            "cv_rmse_mean": forest_rmse_scores.mean(),
        },
    ]
)

model_results = model_results.sort_values("test_rmse")
print(model_results)

model_results.to_csv(METRICS_PATH / "lab_style_model_comparison.csv", index=False)

# %% [markdown]
# ## Task 12: Fine-tune Random Forest with GridSearchCV
#
# We use a small grid because the dataset is small and the project should stay simple.

# %%
param_grid = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": [1.0, "sqrt", 0.5],
}

forest_for_search = Pipeline(
    [
        ("preprocess", make_preprocessor()),
        ("model", RandomForestRegressor(random_state=42, n_jobs=1)),
    ]
)

grid_search = GridSearchCV(
    forest_for_search,
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=1,
)

grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best CV RMSE:", -grid_search.best_score_)

# %% [markdown]
# ## Task 13: Evaluate the final model on the test set

# %%
final_model = grid_search.best_estimator_
final_predictions = final_model.predict(X_test)

final_mae = mean_absolute_error(y_test, final_predictions)
final_rmse = rmse_score(y_test, final_predictions)
final_r2 = r2_score(y_test, final_predictions)

print("Final model MAE:", final_mae)
print("Final model RMSE:", final_rmse)
print("Final model R2:", final_r2)

# %%
residuals = y_test - final_predictions

plt.scatter(final_predictions, residuals, alpha=0.7)
plt.axhline(0, color="black", linewidth=1)
plt.xlabel("Predicted MPG")
plt.ylabel("Residual MPG")
plt.title("Residual plot for final model")
save_fig("final_model_residuals")
plt.close()

# %% [markdown]
# ## Task 14: Analyze the best model
#
# This is similar to the feature importance part in Lab 3.
# Random Forest can tell us which transformed features were most useful.

# %%
feature_importances = final_model["model"].feature_importances_
feature_importances.round(2)

importance_table = pd.DataFrame(
    {
        "feature": final_model["preprocess"].get_feature_names_out(),
        "importance": feature_importances,
    }
).sort_values("importance", ascending=False)

print(importance_table)
importance_table.to_csv(METRICS_PATH / "lab_style_feature_importances.csv", index=False)

# %% [markdown]
# ## Task 15: Save the trained model
#
# We save the complete pipeline, not only the regressor.
# This is important because the pipeline contains both preprocessing and the model.

# %%
MODEL_PATH = MODELS_PATH / "model.joblib"
MODEL_CARD_PATH = MODELS_PATH / "model_card.json"

joblib.dump(final_model, MODEL_PATH)

model_card = {
    "model_version": "rf-lab-style",
    "dataset": "UCI Auto MPG",
    "final_model": "RandomForestRegressor",
    "features": feature_columns,
    "target": "mpg",
    "best_params": grid_search.best_params_,
    "metrics": {
        "test_mae": float(final_mae),
        "test_rmse": float(final_rmse),
        "test_r2": float(final_r2),
    },
    "notes": [
        "Historical Auto MPG model",
        "Use only for vehicles similar to the training data",
    ],
}

MODEL_CARD_PATH.write_text(json.dumps(model_card, indent=2), encoding="utf-8")

print("Saved model to:", MODEL_PATH)
print("Saved model card to:", MODEL_CARD_PATH)


# %% [markdown]
# ## Task 16: Make one prediction

# %%
sample_vehicle = pd.DataFrame(
    [
        {
            "cylinders": 4,          # Motor: 2.0 I4
            "displacement": 122.0,   # 2.0 liter is about 122 cubic inches
            "horsepower": 140.0,     # 140 hk
            "acceleration": 9.8,     # 0-100 km/h, approximate value
            "origin": 2,              # Ford Mondeo built in Europe/Germany
            "weight": 2290.0,
            "model_year": 90,
        }
    ]
)

sample_prediction = final_model.predict(sample_vehicle)
print("Predicted MPG for sample vehicle:", round(sample_prediction[0], 2))
