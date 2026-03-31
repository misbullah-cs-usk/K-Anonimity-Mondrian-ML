#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mondrian_k_anonymity_implementation.py

End-to-end experiment for:
1. Mondrian k-anonymity on the UCI Adult dataset
2. Privacy-utility evaluation using several ML models
3. Two settings:
   - Numeric_QI: only numerical QI are anonymized
   - Mixed_QI: numerical QI are anonymized + categorical QI are globally generalized

Models:
- Logistic Regression
- Random Forest
- SVM
- MLP
- 1D CNN (PyTorch)

Metrics:
- Misclassification Rate
- Accuracy
- Precision
- Recall
- AUC

Usage example:
python adult_mondrian_privacy_ml.py --data_path adult.data --output_dir results

Optional:
python adult_mondrian_privacy_ml.py --data_path adult.data --output_dir results --k_values 2 5 10 20 50 100 --cnn_epochs 10
"""

# =========================
# Section 1: Imports
# =========================
import os
import re
import json
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# =========================
# Section 2: Global config
# =========================
ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

INCOME_MAP = {"<=50K": 0, ">50K": 1}

# Numerical QI used for Mondrian partitioning
QI_NUM = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

# Categorical QI used in mixed-QI setting
QI_CAT = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]

SA_COL = "income"


# =========================
# Section 3: Utility helpers
# =========================
def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    """Save a Python object to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# =========================
# Section 4: Load and clean Adult dataset
# =========================
def load_adult_data(data_path: str) -> pd.DataFrame:
    """
    Load adult.data file, remove missing values, and standardize the income label.
    """
    df = pd.read_csv(
        data_path,
        header=None,
        names=ADULT_COLUMNS,
        na_values=" ?",
        skipinitialspace=True
    )

    # Drop rows with missing values
    df = df.dropna().reset_index(drop=True)

    # Clean string values for income
    df["income"] = df["income"].astype(str).str.strip()

    return df


# =========================
# Section 5: Mondrian for numerical QI
# =========================
def normalized_width(df: pd.DataFrame, col: str, global_min: dict, global_max: dict) -> float:
    """
    Compute normalized width for a numerical column within a partition.
    """
    if global_max[col] == global_min[col]:
        return 0.0

    part_min = df[col].min()
    part_max = df[col].max()
    return (part_max - part_min) / (global_max[col] - global_min[col])


def choose_split_dimension(df: pd.DataFrame, qi_cols: list, global_min: dict, global_max: dict):
    """
    Choose split dimension with largest normalized width.
    """
    widths = {col: normalized_width(df, col, global_min, global_max) for col in qi_cols}
    split_col = max(widths, key=widths.get)
    return split_col, widths


def split_partition(df: pd.DataFrame, split_col: str):
    """
    Split a partition into two child partitions by sorting on split_col and cutting at the median index.
    """
    sorted_df = df.sort_values(by=split_col).reset_index(drop=True)
    mid = len(sorted_df) // 2
    left = sorted_df.iloc[:mid].copy()
    right = sorted_df.iloc[mid:].copy()
    return left, right


def can_split(df: pd.DataFrame, k: int) -> bool:
    """
    A partition can be split only if both child partitions can have at least k records.
    """
    return len(df) >= 2 * k


def mondrian_partition(df: pd.DataFrame, qi_cols: list, k: int, global_min: dict, global_max: dict):
    """
    Recursively partition the data using a basic Mondrian algorithm.
    """
    # Stop if not enough records to split safely
    if not can_split(df, k):
        return [df]

    split_col, widths = choose_split_dimension(df, qi_cols, global_min, global_max)

    # Stop if no meaningful width remains
    if widths[split_col] == 0:
        return [df]

    left, right = split_partition(df, split_col)

    # Stop if split violates k-anonymity
    if len(left) < k or len(right) < k:
        return [df]

    return (
        mondrian_partition(left, qi_cols, k, global_min, global_max) +
        mondrian_partition(right, qi_cols, k, global_min, global_max)
    )


def interval_string(min_val, max_val) -> str:
    """
    Convert a min/max pair to an interval string.
    """
    if min_val == max_val:
        return str(min_val)
    return f"[{min_val}, {max_val}]"


def generalize_partition(df_part: pd.DataFrame, qi_cols: list) -> pd.DataFrame:
    """
    Replace numerical QI values in one partition with interval strings.
    """
    gen_df = df_part.copy()

    for col in qi_cols:
        min_val = df_part[col].min()
        max_val = df_part[col].max()
        gen_value = interval_string(min_val, max_val)
        gen_df[col] = gen_value

    return gen_df


def anonymize_with_mondrian(df: pd.DataFrame, qi_cols: list, k: int):
    """
    Apply Mondrian anonymization to the given numerical QI columns.

    Returns:
    - anon_df: anonymized DataFrame with interval strings on qi_cols
    - partitions: list of final partitions
    """
    df = df.reset_index(drop=True).copy()
    global_min = {col: df[col].min() for col in qi_cols}
    global_max = {col: df[col].max() for col in qi_cols}

    partitions = mondrian_partition(df, qi_cols, k, global_min, global_max)

    anon_parts = [generalize_partition(part, qi_cols) for part in partitions]
    anon_df = pd.concat(anon_parts, axis=0).reset_index(drop=True)

    return anon_df, partitions


def equivalence_class_summary(anon_df: pd.DataFrame, qi_cols: list) -> dict:
    """
    Summarize equivalence class sizes for anonymized data.
    """
    eq_sizes = anon_df.groupby(qi_cols).size().reset_index(name="group_size")
    return {
        "min_equivalence_class_size": int(eq_sizes["group_size"].min()),
        "max_equivalence_class_size": int(eq_sizes["group_size"].max()),
        "avg_equivalence_class_size": float(eq_sizes["group_size"].mean())
    }


# =========================
# Section 6: Categorical generalization for mixed-QI setting
# =========================
def country_to_region(country: str) -> str:
    """
    Map native-country to a broader region.
    """
    if country in ["United-States", "Canada"]:
        return "North-America"
    elif country in ["Mexico"]:
        return "Central-America"
    else:
        return "Other"


def generalize_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply simple taxonomy-style generalization to categorical QI.
    """
    df = df.copy()

    workclass_map = {
        "Private": "Private",
        "Self-emp-not-inc": "Self-employed",
        "Self-emp-inc": "Self-employed",
        "Federal-gov": "Government",
        "Local-gov": "Government",
        "State-gov": "Government",
        "Without-pay": "Other",
        "Never-worked": "Other"
    }

    education_map = {
        "Bachelors": "Higher",
        "Masters": "Higher",
        "Doctorate": "Higher",
        "Prof-school": "Higher",
        "Assoc-acdm": "Higher",
        "Assoc-voc": "Higher",

        "HS-grad": "Secondary",
        "Some-college": "Secondary",

        "12th": "Basic",
        "11th": "Basic",
        "10th": "Basic",
        "9th": "Basic",
        "7th-8th": "Basic",
        "5th-6th": "Basic",
        "1st-4th": "Basic",
        "Preschool": "Basic"
    }

    marital_map = {
        "Married-civ-spouse": "Married",
        "Married-AF-spouse": "Married",
        "Married-spouse-absent": "Married",
        "Never-married": "Single",
        "Divorced": "Separated",
        "Separated": "Separated",
        "Widowed": "Separated"
    }

    occupation_map = {
        "Tech-support": "Tech",
        "Craft-repair": "Manual",
        "Other-service": "Service",
        "Sales": "Sales",
        "Exec-managerial": "Management",
        "Prof-specialty": "Professional",
        "Handlers-cleaners": "Manual",
        "Machine-op-inspct": "Manual",
        "Adm-clerical": "Clerical",
        "Farming-fishing": "Manual",
        "Transport-moving": "Manual",
        "Priv-house-serv": "Service",
        "Protective-serv": "Service",
        "Armed-Forces": "Other"
    }

    # Apply mappings
    df["workclass"] = df["workclass"].map(workclass_map).fillna("Other")
    df["education"] = df["education"].map(education_map).fillna("Other")
    df["marital-status"] = df["marital-status"].map(marital_map).fillna("Other")
    df["occupation"] = df["occupation"].map(occupation_map).fillna("Other")

    # Keep some categorical QI unchanged or lightly generalized
    df["relationship"] = df["relationship"].fillna("Other")
    df["race"] = df["race"].fillna("Other")
    df["sex"] = df["sex"].fillna("Other")
    df["native-country"] = df["native-country"].apply(country_to_region)

    return df


# =========================
# Section 7: Convert anonymized intervals into ML-ready numeric values
# =========================
def parse_interval_to_midpoint(x):
    """
    Convert scalar or interval string to a numeric midpoint.

    Examples:
    39 -> 39.0
    "39" -> 39.0
    "[25, 37]" -> 31.0
    """
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    x = str(x).strip()

    if x.startswith("[") and x.endswith("]"):
        nums = re.findall(r"-?\d+\.?\d*", x)
        if len(nums) == 2:
            a, b = map(float, nums)
            return (a + b) / 2.0

    try:
        return float(x)
    except ValueError:
        return np.nan


def convert_qi_intervals_to_midpoints(df_input: pd.DataFrame, qi_cols: list) -> pd.DataFrame:
    """
    Convert interval-based numerical QI columns to midpoint representation.
    """
    df_out = df_input.copy()
    for col in qi_cols:
        df_out[col] = df_out[col].apply(parse_interval_to_midpoint)
    return df_out


def prepare_dataset_for_ml(df_input: pd.DataFrame, qi_num: list, target_col: str = "income") -> pd.DataFrame:
    """
    Convert anonymized intervals to numeric values and create binary target.
    """
    df_out = df_input.copy()
    df_out = convert_qi_intervals_to_midpoints(df_out, qi_num)
    df_out["income_binary"] = df_out[target_col].map(INCOME_MAP)
    return df_out


def split_X_y(df_input: pd.DataFrame, target_binary_col: str = "income_binary", drop_target_col: str = "income"):
    """
    Split DataFrame into X and y.
    """
    X = df_input.drop(columns=[drop_target_col, target_binary_col])
    y = df_input[target_binary_col].copy()
    return X, y


def build_train_test_dict(df_input: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray):
    """
    Build consistent train/test partitions using fixed row indices.
    """
    X, y = split_X_y(df_input)

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

def make_preprocessor(X_reference: pd.DataFrame) -> ColumnTransformer:
    """
    Build preprocessing pipeline:
    - numeric: median imputation + standard scaling
    - categorical: most_frequent imputation + one-hot encoding
    """
    numeric_features = X_reference.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_reference.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return preprocessor

# =========================
# Section 8: Evaluation metrics
# =========================
def evaluate_classification(y_true, y_pred, y_prob) -> dict:
    """
    Compute required classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    mis = 1.0 - acc
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan

    return {
        "accuracy": float(acc),
        "misclassification_rate": float(mis),
        "precision": float(prec),
        "recall": float(rec),
        "auc": float(auc) if not pd.isna(auc) else np.nan
    }


# =========================
# Section 9: Classical ML models
# =========================
def get_models(random_state: int = 42) -> dict:
    """
    Return all classical ML models used in the experiment.
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            C=1.0,
            gamma="scale"
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=200,
            random_state=random_state
        )
    }


def evaluate_classical_models(dataset_splits: dict, preprocessor: ColumnTransformer, random_state: int = 42) -> pd.DataFrame:
    """
    Train and evaluate classical ML models on all datasets.
    """
    models = get_models(random_state=random_state)
    all_results = []

    for model_name, model in models.items():
        print(f"[Classical] Running model: {model_name}")

        for dataset_name, data in dataset_splits.items():
            X_train = data["X_train"]
            X_test = data["X_test"]
            y_train = data["y_train"]
            y_test = data["y_test"]

            pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("classifier", model)
            ])

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            metrics = evaluate_classification(y_test, y_pred, y_prob)
            metrics["model"] = model_name
            metrics["dataset"] = dataset_name
            metrics["k"] = 0 if dataset_name == "original" else int(dataset_name.split("=")[1])

            all_results.append(metrics)

    return pd.DataFrame(all_results).sort_values(["model", "k"]).reset_index(drop=True)


# =========================
# Section 10: PyTorch 1D CNN
# =========================
class TabularCNNDataset(Dataset):
    """
    PyTorch dataset for tabular data converted to CNN input.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNN1D(nn.Module):
    """
    1D CNN for tabular binary classification.
    Input shape expected by PyTorch Conv1d:
    (batch_size, channels=1, sequence_length=num_features)
    """
    def __init__(self, input_length: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Automatically compute flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            dummy_out = self.features(dummy)
            flattened_dim = dummy_out.view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def transform_with_preprocessor(preprocessor: ColumnTransformer, X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Transform train/test data using sklearn preprocessor and convert to dense float32 arrays.
    """
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    if hasattr(X_train_trans, "toarray"):
        X_train_trans = X_train_trans.toarray()
        X_test_trans = X_test_trans.toarray()

    return X_train_trans.astype(np.float32), X_test_trans.astype(np.float32)


def reshape_for_cnn_pytorch(X: np.ndarray) -> np.ndarray:
    """
    Reshape from (samples, features) to (samples, channels=1, features).
    """
    return X.reshape(X.shape[0], 1, X.shape[1]).astype(np.float32)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train PyTorch CNN for one epoch.
    """
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    return running_loss / len(dataloader.dataset)


def predict_model(model, dataloader, device):
    """
    Run inference and return true labels, hard predictions, and probabilities.
    """
    model.eval()
    all_probs = []
    all_preds = []
    all_true = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)

            logits = model(X_batch)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_true.extend(y_batch.numpy())

    return np.array(all_true), np.array(all_preds), np.array(all_probs)


def train_evaluate_cnn_pytorch(
    data: dict,
    preprocessor: ColumnTransformer,
    device,
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 42
):
    """
    Train and evaluate PyTorch 1D CNN on one dataset split.
    """
    set_seed(seed)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    X_train_trans, X_test_trans = transform_with_preprocessor(preprocessor, X_train, X_test)

    X_train_cnn = reshape_for_cnn_pytorch(X_train_trans)
    X_test_cnn = reshape_for_cnn_pytorch(X_test_trans)

    train_dataset = TabularCNNDataset(X_train_cnn, y_train)
    test_dataset = TabularCNNDataset(X_test_cnn, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_length = X_train_cnn.shape[2]
    model = CNN1D(input_length=input_length).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = []
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss})

    y_true, y_pred, y_prob = predict_model(model, test_loader, device)
    metrics = evaluate_classification(y_true, y_pred, y_prob)

    history_df = pd.DataFrame(history)
    return metrics, history_df


def evaluate_cnn_model(
    dataset_splits: dict,
    preprocessor: ColumnTransformer,
    device,
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    seed: int = 42,
    model_name: str = "CNN_1D_PyTorch"
) -> pd.DataFrame:
    """
    Train and evaluate PyTorch CNN on all datasets.
    """
    cnn_results = []

    for dataset_name, data in dataset_splits.items():
        print(f"[PyTorch CNN] Running on dataset: {dataset_name}")

        metrics, history_df = train_evaluate_cnn_pytorch(
            data=data,
            preprocessor=preprocessor,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed
        )

        metrics["model"] = model_name
        metrics["dataset"] = dataset_name
        metrics["k"] = 0 if dataset_name == "original" else int(dataset_name.split("=")[1])
        cnn_results.append(metrics)

    return pd.DataFrame(cnn_results).sort_values("k").reset_index(drop=True)


# =========================
# Section 11: Build anonymized datasets
# =========================
def create_numeric_qi_anonymized_datasets(df: pd.DataFrame, k_values: list, output_dir: str):
    """
    Create full datasets where only numerical QI are anonymized.
    """
    results = {}
    summary_rows = []

    for k in k_values:
        print(f"[Numeric_QI] Anonymizing with k={k}")

        work_df = df[QI_NUM + [SA_COL]].copy()
        anon_qi_df, partitions = anonymize_with_mondrian(work_df, QI_NUM, k)

        # Replace only numerical QI in the full dataset
        full_df = df.copy().reset_index(drop=True)
        for col in QI_NUM:
            full_df[col] = anon_qi_df[col]

        results[k] = full_df

        # Summaries
        eq_summary = equivalence_class_summary(anon_qi_df, QI_NUM)
        summary_rows.append({
            "setting": "Numeric_QI",
            "k": k,
            "num_partitions": len(partitions),
            "min_partition_size": int(min(len(p) for p in partitions)),
            "max_partition_size": int(max(len(p) for p in partitions)),
            "avg_partition_size": float(np.mean([len(p) for p in partitions])),
            **eq_summary
        })

        # Save anonymized full dataset
        full_df.to_csv(os.path.join(output_dir, f"adult_full_mondrian_numeric_qi_k{k}.csv"), index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "partition_summary_numeric_qi.csv"), index=False)
    return results, summary_df


def create_mixed_qi_anonymized_datasets(df: pd.DataFrame, k_values: list, output_dir: str):
    """
    Create full datasets for mixed-QI setting:
    - categorical QI are globally generalized
    - numerical QI are anonymized with Mondrian
    """
    results = {}
    summary_rows = []

    # First apply categorical generalization globally
    df_cat = generalize_categorical(df)

    for k in k_values:
        print(f"[Mixed_QI] Anonymizing with k={k}")

        work_df = df_cat[QI_NUM + QI_CAT + [SA_COL]].copy()
        anon_qi_df, partitions = anonymize_with_mondrian(work_df, QI_NUM, k)

        # Replace only numerical QI from Mondrian result
        full_df = df_cat.copy().reset_index(drop=True)
        for col in QI_NUM:
            full_df[col] = anon_qi_df[col]

        results[k] = full_df

        eq_summary = equivalence_class_summary(anon_qi_df, QI_NUM)
        summary_rows.append({
            "setting": "Mixed_QI",
            "k": k,
            "num_partitions": len(partitions),
            "min_partition_size": int(min(len(p) for p in partitions)),
            "max_partition_size": int(max(len(p) for p in partitions)),
            "avg_partition_size": float(np.mean([len(p) for p in partitions])),
            **eq_summary
        })

        full_df.to_csv(os.path.join(output_dir, f"adult_full_mondrian_mixed_qi_k{k}.csv"), index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "partition_summary_mixed_qi.csv"), index=False)
    return results, summary_df


# =========================
# Section 12: Build experiment datasets
# =========================
def build_numeric_qi_experiment(df: pd.DataFrame, anonymized_results: dict):
    """
    Build dataset_splits for the Numeric_QI setting:
    original + anonymized versions.
    """
    prepared_original_df = prepare_dataset_for_ml(df, QI_NUM, SA_COL)

    all_indices = np.arange(len(prepared_original_df))
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=42,
        stratify=prepared_original_df["income_binary"]
    )

    dataset_splits = {
        "original": build_train_test_dict(prepared_original_df, train_idx, test_idx)
    }

    for k, df_k in anonymized_results.items():
        prepared_k = prepare_dataset_for_ml(df_k, QI_NUM, SA_COL)
        dataset_splits[f"k={k}"] = build_train_test_dict(prepared_k, train_idx, test_idx)

    # Build preprocessor from original train data
    X_reference = dataset_splits["original"]["X_train"]
    preprocessor = make_preprocessor(X_reference)

    return dataset_splits, preprocessor, train_idx, test_idx


def build_mixed_qi_experiment(anonymized_results: dict, train_idx: np.ndarray, test_idx: np.ndarray):
    """
    Build dataset_splits for the Mixed_QI setting:
    anonymized versions only, using the same train/test indices.
    """
    dataset_splits = {}

    for k, df_k in anonymized_results.items():
        prepared_k = prepare_dataset_for_ml(df_k, QI_NUM, SA_COL)
        dataset_splits[f"k={k}"] = build_train_test_dict(prepared_k, train_idx, test_idx)

    return dataset_splits
# =========================
# Section 15: Plot helpers
# =========================
def save_line_plot(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str, title: str, output_path: str):
    """
    Save a simple line plot as PNG.
    """
    plt.figure(figsize=(10, 6))

    for name in sorted(df[hue_col].dropna().unique()):
        subset = df[df[hue_col] == name].sort_values(x_col)
        plt.plot(subset[x_col], subset[y_col], marker="o", label=str(name))

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_model_comparison_plots(all_results: pd.DataFrame, output_dir: str):
    """
    Save overall and per-model comparison plots as PNG.
    """
    plots_dir = os.path.join(output_dir, "plots")
    ensure_dir(plots_dir)

    # Overall plots by setting
    for setting in sorted(all_results["setting"].dropna().unique()):
        subset = all_results[all_results["setting"] == setting].copy()

        save_line_plot(
            df=subset,
            x_col="k",
            y_col="accuracy",
            hue_col="model",
            title=f"Accuracy vs Privacy Level ({setting})",
            output_path=os.path.join(plots_dir, f"accuracy_vs_k_{setting}.png")
        )

        save_line_plot(
            df=subset,
            x_col="k",
            y_col="auc",
            hue_col="model",
            title=f"AUC vs Privacy Level ({setting})",
            output_path=os.path.join(plots_dir, f"auc_vs_k_{setting}.png")
        )

        save_line_plot(
            df=subset,
            x_col="k",
            y_col="precision",
            hue_col="model",
            title=f"Precision vs Privacy Level ({setting})",
            output_path=os.path.join(plots_dir, f"precision_vs_k_{setting}.png")
        )

        save_line_plot(
            df=subset,
            x_col="k",
            y_col="recall",
            hue_col="model",
            title=f"Recall vs Privacy Level ({setting})",
            output_path=os.path.join(plots_dir, f"recall_vs_k_{setting}.png")
        )

        save_line_plot(
            df=subset,
            x_col="k",
            y_col="misclassification_rate",
            hue_col="model",
            title=f"Misclassification Rate vs Privacy Level ({setting})",
            output_path=os.path.join(plots_dir, f"misclassification_vs_k_{setting}.png")
        )

    # Per-model plots comparing settings
    for model_name in sorted(all_results["model"].dropna().unique()):
        subset = all_results[all_results["model"] == model_name].copy()

        save_line_plot(
            df=subset,
            x_col="k",
            y_col="accuracy",
            hue_col="setting",
            title=f"{model_name}: Accuracy (Numeric_QI vs Mixed_QI)",
            output_path=os.path.join(plots_dir, f"{model_name}_accuracy_numeric_vs_mixed.png")
        )

        save_line_plot(
            df=subset,
            x_col="k",
            y_col="auc",
            hue_col="setting",
            title=f"{model_name}: AUC (Numeric_QI vs Mixed_QI)",
            output_path=os.path.join(plots_dir, f"{model_name}_auc_numeric_vs_mixed.png")
        )


def save_partition_plots(numeric_partition_summary: pd.DataFrame, mixed_partition_summary: pd.DataFrame, output_dir: str):
    """
    Save partition/anonymization summary plots as PNG.
    """
    plots_dir = os.path.join(output_dir, "plots")
    ensure_dir(plots_dir)

    combined = pd.concat(
        [numeric_partition_summary, mixed_partition_summary],
        axis=0
    ).sort_values(["setting", "k"])

    save_line_plot(
        df=combined,
        x_col="k",
        y_col="avg_partition_size",
        hue_col="setting",
        title="Average Partition Size vs k",
        output_path=os.path.join(plots_dir, "avg_partition_size_vs_k.png")
    )

    save_line_plot(
        df=combined,
        x_col="k",
        y_col="min_equivalence_class_size",
        hue_col="setting",
        title="Minimum Equivalence Class Size vs k",
        output_path=os.path.join(plots_dir, "min_equivalence_class_size_vs_k.png")
    )

    save_line_plot(
        df=combined,
        x_col="k",
        y_col="num_partitions",
        hue_col="setting",
        title="Number of Partitions vs k",
        output_path=os.path.join(plots_dir, "num_partitions_vs_k.png")
    )



# =========================
# Section 13: Main experiment runner
# =========================
def run_experiment(args):
    """
    Main pipeline:
    1. Load data
    2. Create anonymized datasets
    3. Build train/test splits
    4. Run classical ML + PyTorch CNN
    5. Save results
    """
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset
    print("Loading Adult dataset...")
    df = load_adult_data(args.data_path)
    print("Cleaned dataset shape:", df.shape)

    # Save cleaned original dataset
    df.to_csv(os.path.join(args.output_dir, "adult_cleaned.csv"), index=False)

    # Create anonymized datasets
    numeric_anonymized_results, numeric_partition_summary = create_numeric_qi_anonymized_datasets(
        df=df,
        k_values=args.k_values,
        output_dir=args.output_dir
    )

    mixed_anonymized_results, mixed_partition_summary = create_mixed_qi_anonymized_datasets(
        df=df,
        k_values=args.k_values,
        output_dir=args.output_dir
    )

    # Build experiment datasets
    numeric_dataset_splits, preprocessor, train_idx, test_idx = build_numeric_qi_experiment(
        df=df,
        anonymized_results=numeric_anonymized_results
    )

    mixed_dataset_splits = build_mixed_qi_experiment(
        anonymized_results=mixed_anonymized_results,
        train_idx=train_idx,
        test_idx=test_idx
    )

    # Run classical models on Numeric_QI
    numeric_classical_results = evaluate_classical_models(
        dataset_splits=numeric_dataset_splits,
        preprocessor=preprocessor,
        random_state=args.seed
    )
    numeric_classical_results["setting"] = "Numeric_QI"

    # Run PyTorch CNN on Numeric_QI
    numeric_cnn_results = evaluate_cnn_model(
        dataset_splits=numeric_dataset_splits,
        preprocessor=preprocessor,
        device=device,
        epochs=args.cnn_epochs,
        batch_size=args.cnn_batch_size,
        learning_rate=args.cnn_learning_rate,
        seed=args.seed,
        model_name="CNN_1D_PyTorch"
    )
    numeric_cnn_results["setting"] = "Numeric_QI"

    # Run classical models on Mixed_QI
    mixed_classical_results = evaluate_classical_models(
        dataset_splits=mixed_dataset_splits,
        preprocessor=preprocessor,
        random_state=args.seed
    )
    mixed_classical_results["setting"] = "Mixed_QI"

    # Run PyTorch CNN on Mixed_QI
    mixed_cnn_results = evaluate_cnn_model(
        dataset_splits=mixed_dataset_splits,
        preprocessor=preprocessor,
        device=device,
        epochs=args.cnn_epochs,
        batch_size=args.cnn_batch_size,
        learning_rate=args.cnn_learning_rate,
        seed=args.seed,
        model_name="CNN_1D_PyTorch"
    )
    mixed_cnn_results["setting"] = "Mixed_QI"

    # Combine all results
    all_results = pd.concat(
        [
            numeric_classical_results,
            numeric_cnn_results,
            mixed_classical_results,
            mixed_cnn_results
        ],
        axis=0
    ).sort_values(["setting", "model", "k"]).reset_index(drop=True)

    # Save results
    numeric_classical_results.to_csv(os.path.join(args.output_dir, "results_numeric_qi_classical.csv"), index=False)
    numeric_cnn_results.to_csv(os.path.join(args.output_dir, "results_numeric_qi_cnn_pytorch.csv"), index=False)
    mixed_classical_results.to_csv(os.path.join(args.output_dir, "results_mixed_qi_classical.csv"), index=False)
    mixed_cnn_results.to_csv(os.path.join(args.output_dir, "results_mixed_qi_cnn_pytorch.csv"), index=False)
    all_results.to_csv(os.path.join(args.output_dir, "all_results.csv"), index=False)

    # Create pivot tables
    accuracy_table = all_results.pivot_table(
        index=["setting", "k"],
        columns="model",
        values="accuracy"
    )
    auc_table = all_results.pivot_table(
        index=["setting", "k"],
        columns="model",
        values="auc"
    )

    accuracy_table.to_csv(os.path.join(args.output_dir, "accuracy_table.csv"))
    auc_table.to_csv(os.path.join(args.output_dir, "auc_table.csv"))

    # Save plots as PNG
    save_model_comparison_plots(all_results, args.output_dir)
    save_partition_plots(numeric_partition_summary, mixed_partition_summary, args.output_dir)

    # Save experiment metadata
    metadata = {
        "data_path": args.data_path,
        "output_dir": args.output_dir,
        "k_values": args.k_values,
        "seed": args.seed,
        "cnn_epochs": args.cnn_epochs,
        "cnn_batch_size": args.cnn_batch_size,
        "cnn_learning_rate": args.cnn_learning_rate,
        "numeric_qi": QI_NUM,
        "categorical_qi": QI_CAT,
        "sensitive_attribute": SA_COL,
        "cleaned_rows": int(len(df)),
        "device": str(device)
    }
    save_json(metadata, os.path.join(args.output_dir, "experiment_metadata.json"))

    # Print summary
    print("\nExperiment completed.")
    print("Output directory:", args.output_dir)
    print("\nTop rows of all_results:")
    print(all_results.head(20))
    print("\nAccuracy table:")
    print(accuracy_table)

# =========================
# Section 14: Argument parser
# =========================
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Mondrian k-anonymity + ML utility evaluation on Adult dataset"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to adult.data"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save outputs"
    )

    parser.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=[2, 5, 10, 20, 50, 100],
        help="List of k values to evaluate"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    parser.add_argument(
        "--cnn_epochs",
        type=int,
        default=10,
        help="Number of epochs for PyTorch CNN"
    )

    parser.add_argument(
        "--cnn_batch_size",
        type=int,
        default=128,
        help="Batch size for PyTorch CNN"
    )

    parser.add_argument(
        "--cnn_learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for PyTorch CNN"
    )

    return parser.parse_args()

# =========================
# Section 16: Script entry point
# =========================
if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
