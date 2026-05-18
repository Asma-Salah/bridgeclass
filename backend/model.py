# model.py
# --------
# This script handles everything ML-related:
# 1. Load and preprocess the dataset
# 2. Train a Random Forest classifier
# 3. Evaluate it properly (F1, confusion matrix, ROC-AUC)
# 4. Save the trained model to disk

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# ------------------------------------------------------------------
# STEP 1: Load the dataset
# ------------------------------------------------------------------
print("=" * 50)
print("BRIDGECLASS — Model Training")
print("=" * 50)

df = pd.read_csv("data/students.csv")
print(f"\n Dataset loaded: {df.shape[0]} students, {df.shape[1]} columns")

# ------------------------------------------------------------------
# STEP 2: Preprocessing
# ------------------------------------------------------------------
# Machine learning models only understand numbers.
# "gender" is text ("M" or "F") — we must convert it to 0 and 1.
# This is called Label Encoding.

le = LabelEncoder()
df["gender_encoded"] = le.fit_transform(df["gender"])
# M → 0, F → 1 (alphabetical order)
print(f"\n Gender encoded: M=0, F=1")

# Drop columns the model should not use:
# - "name" is just an identifier, not a pattern
# - "gender" is replaced by "gender_encoded"
# - "dropped_out" is the TARGET (what we predict, not an input)

FEATURES = [
    "attendance_rate",
    "avg_grade",
    "absences_last_month",
    "gender_encoded",
    "distance_km",
    "num_siblings",
    "fee_arrears"
]

TARGET = "dropped_out"

X = df[FEATURES]   # inputs — what the model sees
y = df[TARGET]     # output — what the model predicts

print(f"\n Features selected: {FEATURES}")
print(f" Target variable: {TARGET}")

# ------------------------------------------------------------------
# STEP 3: Train/Test Split
# ------------------------------------------------------------------
# We split data into:
# - Training set (80%): model learns from this
# - Test set (20%):     we evaluate on this (model never sees it during training)
#
# stratify=y ensures both sets have the same dropout ratio (18.4%)
# Without stratify, we might accidentally put all dropouts in one set.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing = 200 students
    random_state=42,     # reproducibility
    stratify=y           # preserve class ratio in both splits
)

print(f"\n Data split:")
print(f"   Training set: {X_train.shape[0]} students")
print(f"   Test set:     {X_test.shape[0]} students")
print(f"   Dropout rate in train: {y_train.mean()*100:.1f}%")
print(f"   Dropout rate in test:  {y_test.mean()*100:.1f}%")

# ------------------------------------------------------------------
# STEP 4: Train the Random Forest
# ------------------------------------------------------------------
# Key hyperparameters explained:
# n_estimators=200     → build 200 trees and take majority vote
# max_depth=10         → each tree can ask at most 10 yes/no questions
#                        prevents overfitting (memorizing training data)
# min_samples_leaf=4   → each final decision needs at least 4 students
#                        prevents the model from making rules for 1 person
# class_weight="balanced" → compensates for class imbalance (18% vs 82%)
#                           tells the model: missing a dropout is costly
# random_state=42      → reproducibility

print("\n Training Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1            # use all CPU cores — faster training
)

model.fit(X_train, y_train)
print(" Model trained successfully!")

# ------------------------------------------------------------------
# STEP 5: Make Predictions
# ------------------------------------------------------------------
# predict() returns class labels: 0 or 1
# predict_proba() returns probabilities: [prob_stayed, prob_dropout]
# We use probabilities for the risk score (0.0 to 1.0)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # probability of dropout

# ------------------------------------------------------------------
# STEP 6: Evaluate the Model
# ------------------------------------------------------------------
print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_proba)

print(f"\n Core Metrics:")
print(f"   Accuracy:  {accuracy*100:.1f}%  ← misleading for imbalanced data")
print(f"   Precision: {precision*100:.1f}%  ← of flagged students, how many truly at risk")
print(f"   Recall:    {recall*100:.1f}%  ← of actual dropouts, how many we caught ⭐")
print(f"   F1-Score:  {f1*100:.1f}%  ← balance of precision and recall ⭐")
print(f"   ROC-AUC:   {roc_auc:.3f}    ← overall model quality (1.0 = perfect) ⭐")

print(f"\n Full Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Stayed (0)", "Dropped Out (1)"]))

# ------------------------------------------------------------------
# STEP 7: Cross-Validation
# ------------------------------------------------------------------
# A single train/test split might get lucky or unlucky.
# Cross-validation runs 5 different splits and averages the results.
# This proves the model is consistently good, not just lucky once.

print(" Running 5-fold cross-validation...")
cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1")
print(f" Cross-validation F1 scores: {cv_scores.round(3)}")
print(f"   Mean F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# ------------------------------------------------------------------
# STEP 8: Confusion Matrix Plot
# ------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted: Stayed", "Predicted: Dropout"],
    yticklabels=["Actual: Stayed", "Actual: Dropout"]
)
plt.title("Confusion Matrix — BridgeClass Model", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n Confusion matrix saved to plots/confusion_matrix.png")

# Extract and explain the 4 values
tn, fp, fn, tp = cm.ravel()
print(f"\n   True Negatives  (correctly said 'stayed'):   {tn}")
print(f"   False Positives (wrongly flagged as dropout): {fp}")
print(f"   False Negatives (missed actual dropouts):     {fn} ← most costly")
print(f"   True Positives  (correctly caught dropouts):  {tp}")

# ------------------------------------------------------------------
# STEP 9: ROC Curve Plot
# ------------------------------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="#e74c3c", linewidth=2,
         label=f"BridgeClass Model (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--",
         linewidth=1, label="Random baseline (AUC = 0.500)")
plt.fill_between(fpr, tpr, alpha=0.1, color="#e74c3c")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate (Recall)", fontsize=12)
plt.title("ROC Curve — BridgeClass Model", fontsize=13, fontweight="bold")
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("plots/roc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print(" ROC curve saved to plots/roc_curve.png")

# ------------------------------------------------------------------
# STEP 10: Feature Importance Plot
# ------------------------------------------------------------------
importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=True)

plt.figure(figsize=(8, 5))
bars = plt.barh(
    importance_df["feature"],
    importance_df["importance"],
    color="#3498db",
    edgecolor="white"
)
# Add value labels on each bar
for bar, val in zip(bars, importance_df["importance"]):
    plt.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=10)

plt.xlabel("Importance Score", fontsize=12)
plt.title("Feature Importance — What Drives Dropout Risk",
          fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print(" Feature importance saved to plots/feature_importance.png")

print(f"\n Feature Importance Rankings:")
for _, row in importance_df.sort_values("importance", ascending=False).iterrows():
    print(f"   {row['feature']:<25} {row['importance']:.4f}")

# ------------------------------------------------------------------
# STEP 11: Save the Model
# ------------------------------------------------------------------
# joblib saves the entire trained model to a file.
# The FastAPI backend will load this file to make predictions.
# We also save the feature names so the API knows the correct order.

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/bridgeclass_model.pkl")
joblib.dump(FEATURES, "model/feature_names.pkl")

print(f"\n Model saved to model/bridgeclass_model.pkl")
print(f"Feature names saved to model/feature_names.pkl")
print("\n" + "=" * 50)
print("Training complete! ")
print("=" * 50)