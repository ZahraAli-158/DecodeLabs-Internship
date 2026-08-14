"""
Project 2: Data Classification Using AI
DecodeLabs - AI Agent Fellowship 2026
Zahra | BS-AI | The University of Faisalabad

Goal: Build a basic classification model using a small dataset (Iris).
Pipeline: INPUT -> PROCESS -> OUTPUT (IPO Framework, jaisa slides mein diya gaya hai)

Key Requirements covered:
1. Load and understand a dataset
2. Split data into training and testing sets
3. Apply a simple classification algorithm (KNN)
Bonus (from slides): Feature scaling, Elbow method for K, Confusion Matrix, F1 Score
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
)

# =========================================================
# STEP 1 (INPUT): LOAD AND UNDERSTAND THE DATASET
# =========================================================
print("=" * 60)
print("STEP 1: RAW MATERIAL -> THE IRIS BENCHMARK")
print("=" * 60)

iris = load_iris()
X = iris.data                      # features: sepal length/width, petal length/width
y = iris.target                    # labels: 0=setosa, 1=versicolor, 2=virginica
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df["species"] = pd.Categorical.from_codes(y, target_names)

print(f"Samples: {df.shape[0]}  |  Features (Dimensions): {len(feature_names)}  |  Classes: {len(target_names)}")
print("\nClass balance (har class ke samples):")
print(df["species"].value_counts())
print("\nFirst 5 rows (dataset ki jhalak):")
print(df.head())
print("\nBasic statistics:")
print(df.describe())

# =========================================================
# STEP 2 (PROCESS): THE GATEKEEPER RULE -> FEATURE SCALING
# =========================================================
print("\n" + "=" * 60)
print("STEP 2: FEATURE SCALING (StandardScaler: Mean=0, Variance=1)")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Raw data sample (pehli row):        ", np.round(X[0], 2))
print("Scaled data sample (pehli row):      ", np.round(X_scaled[0], 2))
print("Reason: KNN distance-based algorithm hai, isliye sab features ko same scale par lana zaroori hai")
print("        warna bade range wala feature (e.g. petal length in cm) hi decision dominate kar lega.")

# =========================================================
# STEP 3 (PROCESS): STRUCTURAL INTEGRITY -> TRAIN-TEST SPLIT
# =========================================================
print("\n" + "=" * 60)
print("STEP 3: TRAIN-TEST SPLIT (80% Training / 20% Testing, shuffled)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,   # reproducibility ke liye
    stratify=y,        # taake teeno classes proportionally train/test mein split hon
    shuffle=True,
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")
print("Test set model ne training ke waqt bilkul nahi dekha -> yeh 'locked' rehta hai validation ke liye.")

# =========================================================
# STEP 4 (PROCESS): TUNING THE ENGINE -> CHOOSING OPTIMAL K
# =========================================================
print("\n" + "=" * 60)
print("STEP 4: ELBOW METHOD -> BEST K DHOONDNA")
print("=" * 60)
print("Note: K ko choose karne ke liye TEST SET use nahi kar rahe (yeh data-leakage hoga).")
print("      Iske bajaye TRAINING SET par 5-fold Cross-Validation use kar rahe hain,")
print("      taake test set 'locked' rahe sirf final validation ke liye (Step 3 ka principle).")

error_rates = []
k_range = range(1, 21)

for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn_temp, X_train, y_train, cv=5)
    error_rates.append(1 - cv_scores.mean())

# K=1 ko deliberately avoid karte hain kyunke woh noise/overfitting ka classic case hai (slide: "Tuning the Engine")
candidate_ks = [k for k in k_range if k > 1]
best_k = candidate_ks[int(np.argmin([error_rates[k - 1] for k in candidate_ks]))]
print(f"Optimal K (cross-validation, K=1 overfitting risk ke wajah se exclude kiya): K = {best_k}")

plt.figure(figsize=(8, 5))
plt.plot(k_range, error_rates, marker="o", linestyle="--", color="#1f4e79")
plt.axvline(best_k, color="orange", linestyle=":", label=f"Best K = {best_k}")
plt.title("Elbow Method: Error Rate vs K Value")
plt.xlabel("K Value")
plt.ylabel("Error Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("elbow_curve.png", dpi=150)
plt.close()
print("Elbow curve graph save ho gaya -> elbow_curve.png")

# =========================================================
# STEP 5 (PROCESS): THE WORKFLOW -> INSTANTIATE, FIT, PREDICT
# =========================================================
print("\n" + "=" * 60)
print(f"STEP 5: MODEL TRAINING (KNeighborsClassifier, K={best_k})")
print("=" * 60)

model = KNeighborsClassifier(n_neighbors=best_k)   # INSTANTIATE
model.fit(X_train, y_train)                        # FIT (memorize the map)
predictions = model.predict(X_test)                # PREDICT (apply logic)

print("Model successfully trained and predictions generate ho gayi hain.")

# =========================================================
# STEP 6 (OUTPUT): VALIDATION -> CONFUSION MATRIX + F1 SCORE
# =========================================================
print("\n" + "=" * 60)
print("STEP 6: OUTPUT VALIDATION (Accuracy is a lie, hum deeper dekhte hain)")
print("=" * 60)

acc = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="weighted")

print(f"Accuracy:        {acc:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")

print("\nDetailed Classification Report:")
print(classification_report(y_test, predictions, target_names=target_names))

cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix (rows=actual, cols=predicted):")
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title(f"Confusion Matrix (K={best_k})")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()
print("Confusion matrix heatmap save ho gaya -> confusion_matrix.png")

# =========================================================
# FINAL SUMMARY
# =========================================================
print("\n" + "=" * 60)
print("PROJECT 2 COMPLETE - SUMMARY")
print("=" * 60)
print(f"Dataset:        Iris (150 samples, 4 features, 3 classes)")
print(f"Algorithm:      K-Nearest Neighbors (K={best_k})")
print(f"Train/Test:     {X_train.shape[0]}/{X_test.shape[0]} (80/20 split)")
print(f"Accuracy:       {acc:.2%}")
print(f"F1 Score:       {f1:.4f}")
print("Badge Requirement: Complete ✅")
