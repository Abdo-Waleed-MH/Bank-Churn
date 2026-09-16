"""
Train the bank customer churn model and save it for deployment.

Reproduces the pipeline from the original notebook:
  cleaning -> feature engineering -> preprocessing -> tuned Random Forest
Run this once (locally or in CI) to (re)generate model/bank_churn_model.pkl
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "model/bank_churn_model.pkl"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def age_group(age):
        if age < 30:
            return "Young"
        elif age < 45:
            return "Adult"
        elif age < 60:
            return "Middle_Aged"
        return "Senior"

    df["AgeGroup"] = df["Age"].apply(age_group)
    df["IsZeroBalance"] = (df["Balance"] == 0).astype(int)
    df["BalanceSalaryRatio"] = np.log1p(df["Balance"] / (df["EstimatedSalary"] + 1))
    return df


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
    df = df.drop_duplicates()
    df = add_engineered_features(df)

    X = df.drop(columns=["Exited"])
    y = df["Exited"]

    numerical_features = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary",
        "IsZeroBalance", "BalanceSalaryRatio",
    ]
    categorical_features = ["Geography", "Gender", "AgeGroup"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                                   sparse_output=False), categorical_features),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    grid = GridSearchCV(
        estimator=RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
    )
    grid.fit(X_train_processed, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test_processed)
    print("Best params:", grid.best_params_)
    print("Accuracy: ", round(accuracy_score(y_test, y_pred), 3))
    print("Precision:", round(precision_score(y_test, y_pred), 3))
    print("Recall:   ", round(recall_score(y_test, y_pred), 3))
    print("F1-Score: ", round(f1_score(y_test, y_pred), 3))

    import os
    os.makedirs("model", exist_ok=True)
    deployment_package = {
        "preprocessor": preprocessor,
        "model": best_model,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "sklearn_version": __import__("sklearn").__version__,
    }
    joblib.dump(deployment_package, MODEL_PATH, compress=3)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
