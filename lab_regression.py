"""
Module 5 Week A — Lab: Regression & Evaluation

Build and evaluate logistic and linear regression models on the
Petra Telecom customer churn dataset.

Run: python lab_regression.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def load_data(filepath="data/telecom_churn.csv"):
    """Load the telecom churn dataset."""
    path = Path(filepath)
    if not path.exists():
        # Support tests that pass a legacy "starter/data/..." path.
        fallback = Path(__file__).resolve().parent / "data" / path.name
        path = fallback if fallback.exists() else path
    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nChurn distribution:\n", df["churned"].value_counts(normalize=True))

    return df


def split_data(df, target_col, test_size=0.2, random_state=42):
    """Split data into train and test sets with stratification when appropriate."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify = y if y.nunique() <= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    print("Train size:", X_train.shape)
    print("Test size:", X_test.shape)

    if stratify is not None:
        print("Train target mean:", y_train.mean())
        print("Test target mean:", y_test.mean())

    return X_train, X_test, y_train, y_test


def build_logistic_pipeline():
    """Build a Pipeline with StandardScaler and LogisticRegression."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])
    return pipeline


def build_ridge_pipeline():
    """Build a Pipeline with StandardScaler and Ridge regression."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    return pipeline


def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return classification metrics."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }


def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    """Train the pipeline and return regression metrics."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MAE:", mae)
    print("R²:", r2)

    return {
        "mae": mae,
        "r2": r2
    }


def run_cross_validation(pipeline, X_train, y_train, cv=5):
    """Run stratified cross-validation."""
    cv_splitter = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv_splitter,
        scoring="accuracy"
    )

    print("Fold scores:", scores)

    return scores


if __name__ == "__main__":
    df = load_data()
    if df is not None:
        print(f"\nLoaded {len(df)} rows, {df.shape[1]} columns")

        # Classification Task
        numeric_features = [
            "tenure", "monthly_charges", "total_charges",
            "num_support_calls", "senior_citizen",
            "has_partner", "has_dependents"
        ]

        df_cls = df[numeric_features + ["churned"]].dropna()

        split = split_data(df_cls, "churned")
        if split:
            X_train, X_test, y_train, y_test = split

            pipe = build_logistic_pipeline()
            if pipe:
                metrics = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)
                print(f"\nLogistic Regression Metrics: {metrics}")

                scores = run_cross_validation(pipe, X_train, y_train)
                if scores is not None:
                    print(f"CV Mean: {scores.mean():.3f} +/- {scores.std():.3f}")


        # Regression Task
        df_reg = df[[
            "tenure", "total_charges", "num_support_calls",
            "senior_citizen", "has_partner", "has_dependents",
            "monthly_charges"
        ]].dropna()

        split_reg = split_data(df_reg, "monthly_charges")
        if split_reg:
            X_tr, X_te, y_tr, y_te = split_reg

            ridge_pipe = build_ridge_pipeline()
            if ridge_pipe:
                reg_metrics = evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)
                print(f"\nRidge Regression Metrics: {reg_metrics}")

"""
Summary of findings:

The logistic regression model has low precision but decent recall, which means it catches some churners but also makes many false predictions. The accuracy (~63%) is not very reliable due to class imbalance, and overall the model performance is moderate.

Cross-validation results are consistent (~0.60), showing the model is stable but not very strong.

For regression, Ridge performed well with an r^2 of about 0.71 and reasonable MAE, so it fits the data fairly well.

To improve, I would try tuning the model and testing other algorithms.
"""