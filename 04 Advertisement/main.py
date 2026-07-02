import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================================
# SETTINGS
# =====================================================

plt.style.use("ggplot")

sns.set_theme(
    style="whitegrid",
    font_scale=1.1
)

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("Advertising.csv")

# Remove unwanted index column if present

if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)

print("\nFirst 5 Rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistics:")
print(df.describe())

# =====================================================
# VISUALIZATION 1
# SALES DISTRIBUTION
# =====================================================

plt.figure(figsize=(10,6))

sns.histplot(
    df["Sales"],
    bins=20,
    kde=True,
    label="Sales Distribution"
)

plt.title(
    "Distribution of Sales",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.legend()

plt.tight_layout()
plt.show()

# =====================================================
# VISUALIZATION 2
# CORRELATION HEATMAP
# =====================================================

plt.figure(figsize=(8,6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title(
    "Feature Correlation Heatmap",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()
plt.show()

# =====================================================
# VISUALIZATION 3
# TV ADVERTISING VS SALES
# =====================================================

plt.figure(figsize=(10,6))

sns.regplot(
    data=df,
    x="TV",
    y="Sales",
    scatter_kws={"alpha":0.7},
    line_kws={"label":"Regression Line"}
)

plt.title(
    "TV Advertising Budget vs Sales",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("TV Advertising Spend")
plt.ylabel("Sales")

plt.legend()

plt.tight_layout()
plt.show()

# =====================================================
# VISUALIZATION 4
# RADIO ADVERTISING VS SALES
# =====================================================

plt.figure(figsize=(10,6))

sns.regplot(
    data=df,
    x="Radio",
    y="Sales",
    scatter_kws={"alpha":0.7},
    line_kws={"label":"Regression Line"}
)

plt.title(
    "Radio Advertising Budget vs Sales",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Radio Advertising Spend")
plt.ylabel("Sales")

plt.legend()

plt.tight_layout()
plt.show()

# =====================================================
# VISUALIZATION 5
# NEWSPAPER VS SALES
# =====================================================

plt.figure(figsize=(10,6))

sns.regplot(
    data=df,
    x="Newspaper",
    y="Sales",
    scatter_kws={"alpha":0.7},
    line_kws={"label":"Regression Line"}
)

plt.title(
    "Newspaper Advertising Budget vs Sales",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Newspaper Advertising Spend")
plt.ylabel("Sales")

plt.legend()

plt.tight_layout()
plt.show()

# =====================================================
# VISUALIZATION 6
# PAIRPLOT
# =====================================================

sns.pairplot(
    df,
    diag_kind="hist"
)

plt.show()

# =====================================================
# FEATURES & TARGET
# =====================================================

X = df.drop("Sales", axis=1)

y = df["Sales"]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# MODEL TRAINING
# =====================================================

model = LinearRegression()

model.fit(X_train, y_train)

# =====================================================
# PREDICTION
# =====================================================

y_pred = model.predict(X_test)

# =====================================================
# MODEL EVALUATION
# =====================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n")
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"MAE       : {mae:.3f}")
print(f"MSE       : {mse:.3f}")
print(f"RMSE      : {rmse:.3f}")
print(f"R² SCORE  : {r2:.4f}")

# =====================================================
# VISUALIZATION 7
# ACTUAL VS PREDICTED
# =====================================================

plt.figure(figsize=(10,6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.8,
    label="Predictions"
)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction"
)

plt.title(
    "Actual vs Predicted Sales",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")

plt.legend()

plt.tight_layout()
plt.show()

# =====================================================
# VISUALIZATION 8
# FEATURE IMPORTANCE
# =====================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

importance["Abs"] = abs(
    importance["Coefficient"]
)

importance = importance.sort_values(
    by="Abs",
    ascending=False
)

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance,
    x="Coefficient",
    y="Feature"
)

plt.title(
    "Feature Importance (Linear Regression)",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("Coefficient Value")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()

print("\nFeature Importance:")
print(
    importance[
        ["Feature","Coefficient"]
    ]
)

# =====================================================
# SAMPLE PREDICTION
# =====================================================

sample = pd.DataFrame({
    "TV":[230],
    "Radio":[37],
    "Newspaper":[69]
})

predicted_sales = model.predict(sample)

print("\nSample Prediction")
print("-" * 30)

print(
    f"Predicted Sales: "
    f"{predicted_sales[0]:.2f}"
)

print("\nProject Completed Successfully!")