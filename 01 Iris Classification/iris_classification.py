import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =====================================================
# PROJECT HEADER
# =====================================================

print("=" * 70)
print("IRIS FLOWER CLASSIFICATION USING MACHINE LEARNING")
print("=" * 70)

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("Iris.csv")

print("\nDATASET OVERVIEW")
print("-" * 70)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# =====================================================
# DATA CLEANING
# =====================================================

if "Id" in df.columns:
    df.drop("Id", axis=1, inplace=True)

print("\nData Cleaning Completed Successfully.")

# =====================================================
# SPECIES DISTRIBUTION
# =====================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Species"
)

plt.title(
    "Distribution of Iris Species",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Species")
plt.ylabel("Count")

plt.grid(axis='y', alpha=0.3)

plt.show()

print("\nSPECIES DISTRIBUTION INSIGHTS")
print("-" * 70)

species_counts = df["Species"].value_counts()

for species, count in species_counts.items():
    print(f"{species}: {count} flowers")

if species_counts.nunique() == 1:
    print("\nDataset is perfectly balanced.")
else:
    print("\nDataset is imbalanced.")

# =====================================================
# PAIRPLOT
# =====================================================

print("\nGenerating Pairplot...")

sns.pairplot(
    df,
    hue="Species",
    diag_kind="hist"
)

plt.show()

print("\nPAIRPLOT INSIGHTS")
print("-" * 70)

print("Setosa is usually clearly separated.")
print("Versicolor and Virginica show slight overlap.")
print("Petal features provide the best separation.")

# =====================================================
# CORRELATION HEATMAP
# =====================================================

plt.figure(figsize=(8, 6))

corr_matrix = df.drop("Species", axis=1).corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title(
    "Feature Correlation Heatmap",
    fontsize=15,
    fontweight="bold"
)

plt.show()

print("\nCORRELATION INSIGHTS")
print("-" * 70)

highest_corr = corr_matrix.unstack()
highest_corr = highest_corr[highest_corr != 1]

max_corr = highest_corr.abs().idxmax()

print(
    f"Strongest relationship: "
    f"{max_corr[0]} and {max_corr[1]}"
)

# =====================================================
# FEATURES & TARGET
# =====================================================

X = df.drop("Species", axis=1)
y = df["Species"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# =====================================================
# MODEL TRAINING
# =====================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed.")

# =====================================================
# PREDICTIONS
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# EVALUATION
# =====================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nMODEL PERFORMANCE")
print("-" * 70)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title(
    "Confusion Matrix",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

print("\nCONFUSION MATRIX INSIGHTS")
print("-" * 70)

correct_predictions = np.trace(cm)

print(
    f"Correct Predictions: "
    f"{correct_predictions}/{len(y_test)}"
)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(9, 5))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title(
    "Feature Importance",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Importance Score")
plt.ylabel("Feature")

plt.show()

print("\nFEATURE IMPORTANCE INSIGHTS")
print("-" * 70)

for _, row in importance_df.iterrows():
    print(
        f"{row['Feature']}: "
        f"{row['Importance']:.4f}"
    )

top_feature = importance_df.iloc[0]["Feature"]

print(
    f"\nMost Important Feature: "
    f"{top_feature}"
)

# =====================================================
# SAMPLE PREDICTION
# =====================================================

sample_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(
    sample_flower
)

print("\nSAMPLE PREDICTION")
print("-" * 70)

print(
    "Flower Measurements:"
)

print(
    "Sepal Length = 5.1, "
    "Sepal Width = 3.5, "
    "Petal Length = 1.4, "
    "Petal Width = 0.2"
)

print(
    f"Predicted Species: "
    f"{prediction[0]}"
)

# =====================================================
# FINAL CONCLUSION
# =====================================================

print("\nFINAL PROJECT SUMMARY")
print("=" * 70)

print(f"Dataset Size: {len(df)} flowers")
print(f"Number of Features: {X.shape[1]}")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"Most Important Feature: {top_feature}")

if accuracy >= 0.95:
    print(
        "\nExcellent model performance."
    )
elif accuracy >= 0.85:
    print(
        "\nVery good model performance."
    )
else:
    print(
        "\nModel can be improved."
    )

print("\nProject Completed Successfully.")
print("=" * 70)