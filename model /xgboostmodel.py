
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# 1. Paths
DATA_PATH = Path("data/processed_windows.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# 2. Select 22 CSI-based features
FEATURES = [
    "temporal_std_mean",
    "temporal_std_median",
    "temporal_std_p90",
    "temporal_std_max",
    "delta_mean",
    "delta_p90",
    "delta_std",
    "rowchange_std",
    "profile_drift",
    "range_mean",
]

for i in range(1, 7):
    FEATURES.extend([
        f"bin{i}_std",
        f"bin{i}_delta"
    ])


# 3. Load dataset
df = pd.read_csv(DATA_PATH)

required = set(FEATURES + ["state", "label", "start_s"])
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

if df[FEATURES].isna().any().any():
    raise ValueError("Features contain missing values")

if not np.isfinite(df[FEATURES].to_numpy()).all():
    raise ValueError("Features contain infinite values")

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["label"].value_counts())


# 4. Time-based train/test split
# Avoid splitting overlapping windows randomly.
# This is a preliminary test on the same sessions.

train_parts = []
test_parts = []

for state, group in df.groupby("state"):

    group = group.sort_values("start_s").reset_index(drop=True)

    n = len(group)
    train_end = int(n * 0.70)

    # Skip 2 windows between train and test.
    # Windows are 6 seconds long, 3-second step.
    test_start = train_end + 2

    if n - test_start < 3:
        raise ValueError(
            f"Not enough windows for state: {state}"
        )

    train_parts.append(group.iloc[:train_end])
    test_parts.append(group.iloc[test_start:])

train_df = pd.concat(train_parts, ignore_index=True)
test_df = pd.concat(test_parts, ignore_index=True)

X_train = train_df[FEATURES]
y_train = train_df["label"].astype(int)

X_test = test_df[FEATURES]
y_test = test_df["label"].astype(int)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 5. Define XGBoost classifier
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=100,
    max_depth=2,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    reg_lambda=5,
    scale_pos_weight=negative / positive,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=2
)

# 6. Train model
model.fit(X_train, y_train)

print("\nModel training completed!")


# 7. Make predictions
probabilities = model.predict_proba(X_test)[:, 1]

predictions = (probabilities >= 0.5).astype(int)


# 8. Evaluate
metrics = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "balanced_accuracy": float(
        balanced_accuracy_score(y_test, predictions)
    ),
    "precision": float(
        precision_score(y_test, predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, predictions, zero_division=0)
    ),
    "f1": float(
        f1_score(y_test, predictions, zero_division=0)
    ),
    "macro_f1": float(
        f1_score(y_test, predictions, average="macro")
    ),
}

print("\n=== XGBOOST RESULTS ===")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["NOT MOVING", "MOVING"],
        zero_division=0
    )
)


# 9. Save trained model and feature names
model.save_model(MODEL_DIR / "xgboost_motion.json")

with open(MODEL_DIR / "feature_columns.json", "w") as file:
    json.dump(FEATURES, file, indent=2)

with open(MODEL_DIR / "metrics.json", "w") as file:
    json.dump(metrics, file, indent=2)


# 10. Save test predictions for Streamlit
results = test_df[
    ["state", "start_s", "end_s", "label"]
].copy()

results["predicted_label"] = predictions
results["movement_probability"] = probabilities

results.to_csv(
    MODEL_DIR / "test_predictions.csv",
    index=False
)


# 11. Plot confusion matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

fig, ax = plt.subplots(figsize=(6, 5))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["NOT MOVING", "MOVING"]
).plot(ax=ax, cmap="Blues", values_format="d")

plt.title("XGBoost Confusion Matrix")
plt.tight_layout()

plt.savefig(
    MODEL_DIR / "confusion_matrix.png",
    dpi=200
)
plt.close()


# 12. Plot feature importance
importance = pd.Series(
    model.feature_importances_,
    index=FEATURES
).sort_values(ascending=False)

importance.to_csv(
    MODEL_DIR / "feature_importance.csv",
    header=["importance"]
)

fig, ax = plt.subplots(figsize=(8, 5))

importance.head(10).sort_values().plot.barh(ax=ax)

ax.set_title("Top 10 CSI Features")
ax.set_xlabel("XGBoost Feature Importance")

plt.tight_layout()

plt.savefig(
    MODEL_DIR / "feature_importance.png",
    dpi=200
)
plt.close()

print("\nAll model files saved to:", MODEL_DIR)
