from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys
import pandas as pd
import mlflow
import joblib
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# --- Configuration ---
MLFLOW_TRACKING_URI = "http://35.223.244.50:5000/"
MODEL_NAME = "iris-random-forest"
RUN_NAME = "Random Forest Hyperparameter Search"
MODEL_DOWNLOAD_PATH = "downloaded_models/random_forest_model"  # Adjusted path

app = FastAPI(title="Iris RandomForest API", version="1.1")

# ---------------------------- #
#   Utility / Training Helpers #
# ---------------------------- #
def prepare_data():
    """Loads and splits the Iris dataset."""
    print("Preparing data...")
    data = pd.read_csv("./data.csv")
    data = pd.DataFrame(
        data,
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"],
    )

    train, test = train_test_split(
        data, test_size=0.2, stratify=data["species"], random_state=42
    )

    X_train = train[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y_train = train["species"]
    X_test = test[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y_test = test["species"]

    return X_train, y_train, X_test, y_test


def tune_random_forest(X_train, y_train, X_test, y_test):
    """Train a RandomForest model with hyperparameter tuning and log to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    rf_param_grid = {
        "n_estimators": [50, 100, 200],
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [3, 5, 10],
        "class_weight": [None, "balanced"],
    }

    with mlflow.start_run(run_name=RUN_NAME):
        rf_model = RandomForestClassifier(random_state=42)
        rf_grid_search = GridSearchCV(
            rf_model, rf_param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=2
        )

        rf_grid_search.fit(X_train, y_train)

        best_score_cv = rf_grid_search.best_score_
        test_score = rf_grid_search.score(X_test, y_test)

        mlflow.log_params(rf_grid_search.best_params_)
        mlflow.log_metric("best_cv_accuracy", best_score_cv)
        mlflow.log_metric("final_test_accuracy", test_score)
        mlflow.sklearn.log_model(
            rf_grid_search.best_estimator_,
            "random_forest_model",
            registered_model_name=MODEL_NAME,
        )

        return {
            "best_params": rf_grid_search.best_params_,
            "cv_accuracy": best_score_cv,
            "test_accuracy": test_score,
        }


# ---------------------------- #
#        Fetch Helpers         #
# ---------------------------- #
def fetch_latest_model(client, name):
    """Fetch latest model version details from MLflow Model Registry."""
    versions = client.search_model_versions(
        filter_string=f"name='{name}'", order_by=["version_number DESC"], max_results=1
    )

    if not versions:
        raise ValueError(f"No versions found for model '{name}'")

    latest = versions[0]
    return {
        "version": latest.version,
        "run_id": latest.run_id,
        "status": latest.current_stage,
    }


def download_model(run_id, save_path):
    """Download a specific model version artifacts."""
    downloaded_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="random_forest_model", dst_path=save_path
    )
    return downloaded_path


# ---------------------------- #
#         FastAPI Routes       #
# ---------------------------- #
@app.get("/")
def root():
    return {"message": "Welcome to the Iris RandomForest FastAPI service!"}


@app.post("/train")
def train_model():
    """Train and log the RandomForest model to MLflow."""
    try:
        X_train, y_train, X_test, y_test = prepare_data()
        results = tune_random_forest(X_train, y_train, X_test, y_test)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fetch-latest")
def fetch_model():
    """Fetch metadata of the latest registered model."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        model_info = fetch_latest_model(client, MODEL_NAME)
        return {"status": "success", "model_info": model_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/download")
def download_latest_model():
    """Download the latest model artifact from MLflow."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        version = fetch_latest_model(client, MODEL_NAME)
        downloaded_path = download_model(version["run_id"], MODEL_DOWNLOAD_PATH)
        return {"status": "success", "path": downloaded_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------- #
#        Prediction Route      #
# ---------------------------- #
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict")
def predict_iris(features: IrisFeatures):
    """Predict the Iris species using the downloaded model."""
    try:
        model_path = os.path.join(MODEL_DOWNLOAD_PATH, "model.pkl")

        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model file not found. Please download it first.")

        model = joblib.load(model_path)
        data = [[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ]]

        prediction = model.predict(data)[0]
        return {"status": "success", "predicted_species": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
