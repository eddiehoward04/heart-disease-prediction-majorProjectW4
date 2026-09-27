"""
Heart Disease Prediction - Full ML Pipeline
Week 4 Capstone Project

Dataset: Kaggle - Heart Disease UCI
https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data

Setup:
    1. Download the dataset CSV from the link above.
    2. Save it as data/heart.csv (this exact path/name).
    3. Install dependencies:  pip install -r requirements.txt
    4. Run:  python main.py

Workflow implemented:
    - Data cleaning + feature scaling
    - Train Logistic Regression / Decision Tree
    - Evaluate accuracy, precision, recall
    - Confusion matrix & ROC curve visualization
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
)
import joblib

DATA_PATH = os.path.join("data", "heart.csv")
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"


def load_data(path=DATA_PATH):
    """Load the raw CSV, with a helpful error if it hasn't been downloaded yet."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download it from Kaggle: "
            "https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data\n"
            "and save it as data/heart.csv"
        )
    return pd.read_csv(path)


def clean_data(df):
    """Standardize columns, fix the target label, and handle missing values."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Different releases of this dataset name the label column differently.
    target_col = next(
        (c for c in ["target", "num", "heartdisease", "condition"] if c in df.columns),
        None,
    )
    if target_col is None:
        raise ValueError("Could not find a target/label column in the dataset.")

    # Some versions use 0-4 severity; collapse to binary (0 = no disease, 1 = disease)
    df[target_col] = (df[target_col].astype(int) > 0).astype(int)
    if target_col != "target":
        df = df.rename(columns={target_col: "target"})

    df = df.dropna(how="all")
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    for col in ["id", "unnamed: 0"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def split_and_scale(df):
    """Encode categoricals, split train/test, and scale features."""
    X = df.drop(columns=["target"])
    y = df["target"]
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_models(X_train, y_train):
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)

    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)

    return {"Logistic Regression": log_reg, "Decision Tree": dt}


def evaluate_model(name, model, X_test, y_test):
    """Compute metrics and save a confusion matrix + ROC curve plot for one model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n--- {name} ---")
    for k, v in metrics.items():
        print(f"{k.capitalize():10s}: {v:.3f}")

    slug = name.replace(" ", "_").lower()

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {name}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{slug}.png"))
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"roc_curve_{slug}.png"))
    plt.close(fig)

    return metrics


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data()

    print("Cleaning data...")
    df = clean_data(df)

    print("Splitting and scaling features...")
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df)

    print("Training models...")
    models = train_models(X_train, y_train)

    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(name, model, X_test, y_test)
        slug = name.replace(" ", "_").lower()
        joblib.dump(model, os.path.join(MODEL_DIR, f"{slug}.joblib"))

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    print("\n=== Summary ===")
    summary_df = pd.DataFrame(results).T
    print(summary_df.round(3))
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"))

    print(f"\nPlots saved to '{OUTPUT_DIR}/'. Trained models saved to '{MODEL_DIR}/'.")


if __name__ == "__main__":
    main()
