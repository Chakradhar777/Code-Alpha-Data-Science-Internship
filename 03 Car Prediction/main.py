import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# GLOBAL STYLING
# =====================================================

plt.style.use("ggplot")

sns.set_theme(
    style="whitegrid",
    font_scale=1.1
)

# =====================================================
# PROJECT HEADER
# =====================================================

print("=" * 80)
print("CAR PRICE PREDICTION USING MACHINE LEARNING")
print("=" * 80)

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("car data.csv")

print("\nDATASET OVERVIEW")
print("-" * 80)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# =====================================================
# FEATURE ENGINEERING
# =====================================================

CURRENT_YEAR = 2025

df["Car_Age"] = (
    CURRENT_YEAR -
    df["Year"]
)

print("\nFeature Engineering Completed")

print(
    "\nNew Feature Added: Car_Age"
)

print(
    "Car_Age = Current Year - Manufacturing Year"
)

# =====================================================
# OVERALL DATASET REPORT
# =====================================================

print("\nDATASET INSIGHTS")
print("-" * 80)

avg_price = df["Selling_Price"].mean()

max_price = df["Selling_Price"].max()

min_price = df["Selling_Price"].min()

avg_age = df["Car_Age"].mean()

print(
    f"Average Selling Price : "
    f"{avg_price:.2f} Lakhs"
)

print(
    f"Highest Selling Price : "
    f"{max_price:.2f} Lakhs"
)

print(
    f"Lowest Selling Price : "
    f"{min_price:.2f} Lakhs"
)

print(
    f"Average Car Age : "
    f"{avg_age:.2f} Years"
)

# =====================================================
# GRAPH 1
# SELLING PRICE DISTRIBUTION
# =====================================================

plt.figure(figsize=(12,6))

sns.histplot(
    data=df,
    x="Selling_Price",
    bins=25,
    kde=True,
    label="Selling Price Distribution"
)

plt.title(
    "Distribution of Car Selling Prices",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Selling Price (Lakhs)"
)

plt.ylabel(
    "Frequency"
)

plt.legend()

plt.tight_layout()

plt.show()

print("\nGRAPH 1 INSIGHTS")
print("-" * 80)

median_price = df[
    "Selling_Price"
].median()

print(
    f"Average Price : "
    f"{avg_price:.2f} Lakhs"
)

print(
    f"Median Price  : "
    f"{median_price:.2f} Lakhs"
)

print(
    f"Maximum Price : "
    f"{max_price:.2f} Lakhs"
)

print(
    f"Minimum Price : "
    f"{min_price:.2f} Lakhs"
)

if avg_price > median_price:
    print(
        "\nThe distribution is positively skewed."
    )

    print(
        "A few expensive cars increase the average."
    )

else:
    print(
        "\nThe distribution is approximately symmetric."
    )

# =====================================================
# GRAPH 2
# FUEL TYPE ANALYSIS
# =====================================================

fuel_counts = (
    df["Fuel_Type"]
    .value_counts()
)

plt.figure(figsize=(10,6))

plt.pie(
    fuel_counts.values,
    labels=fuel_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title(
    "Fuel Type Distribution",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()

plt.show()

print("\nGRAPH 2 INSIGHTS")
print("-" * 80)

for fuel, count in fuel_counts.items():

    percentage = (
        count /
        len(df)
    ) * 100

    print(
        f"{fuel:<10} : "
        f"{count} Cars "
        f"({percentage:.2f}%)"
    )

print()

dominant_fuel = fuel_counts.idxmax()

print(
    f"Most Common Fuel Type: "
    f"{dominant_fuel}"
)

print(
    "This indicates customer preference "
    "and market availability."
)

# =====================================================
# PART 1 SUMMARY
# =====================================================

print("\nPART 1 SUMMARY")
print("=" * 80)

print(
    f"Total Cars Analysed : "
    f"{len(df)}"
)

print(
    f"Average Selling Price : "
    f"{avg_price:.2f} Lakhs"
)

print(
    f"Average Car Age : "
    f"{avg_age:.2f} Years"
)

print(
    f"Most Common Fuel Type : "
    f"{dominant_fuel}"
)

print("=" * 80)



# =====================================================
# GRAPH 3
# TRANSMISSION TYPE ANALYSIS
# =====================================================

transmission_counts = (
    df["Transmission"]
    .value_counts()
)

plt.figure(figsize=(10,6))

sns.countplot(
    data=df,
    x="Transmission",
    hue="Transmission",
    legend=False
)

plt.title(
    "Transmission Type Distribution",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel("Transmission Type")
plt.ylabel("Number of Cars")

plt.tight_layout()
plt.show()

print("\nGRAPH 3 INSIGHTS")
print("-" * 80)

for transmission, count in transmission_counts.items():

    percentage = (
        count / len(df)
    ) * 100

    print(
        f"{transmission:<10}: "
        f"{count} Cars "
        f"({percentage:.2f}%)"
    )

print()

most_common_transmission = (
    transmission_counts.idxmax()
)

print(
    f"Most Popular Transmission: "
    f"{most_common_transmission}"
)

if most_common_transmission == "Manual":
    print(
        "Manual cars dominate the dataset,"
    )
    print(
        "indicating affordability and popularity."
    )
else:
    print(
        "Automatic vehicles dominate the market."
    )

# =====================================================
# GRAPH 4
# CAR AGE VS SELLING PRICE
# =====================================================

plt.figure(figsize=(12,7))

sns.scatterplot(
    data=df,
    x="Car_Age",
    y="Selling_Price",
    hue="Fuel_Type",
    s=90
)

plt.title(
    "Car Age vs Selling Price",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel("Car Age (Years)")
plt.ylabel("Selling Price (Lakhs)")

plt.legend(
    title="Fuel Type"
)

plt.tight_layout()
plt.show()

print("\nGRAPH 4 INSIGHTS")
print("-" * 80)

age_price_corr = (
    df["Car_Age"]
    .corr(df["Selling_Price"])
)

print(
    f"Correlation: "
    f"{age_price_corr:.3f}"
)

print()

if age_price_corr < -0.7:
    print(
        "Strong negative relationship detected."
    )

    print(
        "Older cars generally sell for less."
    )

elif age_price_corr < -0.4:
    print(
        "Moderate negative relationship detected."
    )

else:
    print(
        "Weak relationship detected."
    )

print()

oldest_car = df["Car_Age"].max()

newest_car = df["Car_Age"].min()

print(
    f"Oldest Car : {oldest_car} Years"
)

print(
    f"Newest Car : {newest_car} Years"
)

print()

print(
    "This graph demonstrates vehicle depreciation."
)

# =====================================================
# GRAPH 5
# PRESENT PRICE VS SELLING PRICE
# =====================================================

plt.figure(figsize=(12,7))

sns.scatterplot(
    data=df,
    x="Present_Price",
    y="Selling_Price",
    hue="Transmission",
    s=90
)

sns.regplot(
    data=df,
    x="Present_Price",
    y="Selling_Price",
    scatter=False,
    label="Trend Line"
)

plt.title(
    "Present Price vs Selling Price",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel("Current Market Price (Lakhs)")
plt.ylabel("Selling Price (Lakhs)")

plt.legend()

plt.tight_layout()
plt.show()

print("\nGRAPH 5 INSIGHTS")
print("-" * 80)

price_corr = (
    df["Present_Price"]
    .corr(df["Selling_Price"])
)

print(
    f"Correlation: "
    f"{price_corr:.3f}"
)

print()

if price_corr > 0.8:
    print(
        "Very strong positive relationship."
    )

    print(
        "Current market value is a major factor "
        "in determining selling price."
    )

elif price_corr > 0.5:
    print(
        "Moderate positive relationship."
    )

else:
    print(
        "Weak relationship."
    )

print()

avg_present_price = (
    df["Present_Price"]
    .mean()
)

avg_selling_price = (
    df["Selling_Price"]
    .mean()
)

depreciation = (
    (
        avg_present_price -
        avg_selling_price
    )
    / avg_present_price
) * 100

print(
    f"Average Present Price : "
    f"{avg_present_price:.2f} Lakhs"
)

print(
    f"Average Selling Price : "
    f"{avg_selling_price:.2f} Lakhs"
)

print(
    f"Average Depreciation : "
    f"{depreciation:.2f}%"
)

print()

print(
    "Depreciation represents the reduction "
    "in vehicle value over time."
)

# =====================================================
# MARKET ANALYSIS REPORT
# =====================================================

print("\nMARKET ANALYSIS REPORT")
print("=" * 80)

print(
    f"Most Popular Fuel Type      : "
    f"{dominant_fuel}"
)

print(
    f"Most Popular Transmission   : "
    f"{most_common_transmission}"
)

print(
    f"Average Vehicle Age         : "
    f"{avg_age:.2f} Years"
)

print(
    f"Age-Price Correlation       : "
    f"{age_price_corr:.3f}"
)

print(
    f"Present-Selling Correlation : "
    f"{price_corr:.3f}"
)

print(
    f"Average Depreciation        : "
    f"{depreciation:.2f}%"
)

print("=" * 80)



from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================================
# GRAPH 6
# CORRELATION HEATMAP
# =====================================================

numeric_df = df.select_dtypes(include=np.number)

corr_matrix = numeric_df.corr()

plt.figure(figsize=(10,8))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="RdYlBu_r",
    linewidths=0.5,
    fmt=".2f"
)

plt.title(
    "Correlation Heatmap",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()
plt.show()

print("\nGRAPH 6 INSIGHTS")
print("-" * 80)

selling_price_corr = (
    corr_matrix["Selling_Price"]
    .sort_values(ascending=False)
)

print(
    "Features Most Related To Selling Price:\n"
)

for feature, value in selling_price_corr.items():

    if feature != "Selling_Price":

        print(
            f"{feature:<20}: "
            f"{value:.3f}"
        )

print()

best_feature = (
    selling_price_corr.index[1]
)

print(
    f"Strongest Predictor: "
    f"{best_feature}"
)

print(
    "This feature will have significant "
    "influence on price prediction."
)

# =====================================================
# PREPARING DATA
# =====================================================

print("\nDATA PREPROCESSING")
print("=" * 80)

model_df = df.copy()

model_df.drop(
    "Car_Name",
    axis=1,
    inplace=True
)

model_df.drop(
    "Year",
    axis=1,
    inplace=True
)

print(
    "Removed Non-Numeric Identifier: Car_Name"
)

print(
    "Removed Year Column "
    "(Car_Age already created)"
)

# =====================================================
# ONE HOT ENCODING
# =====================================================

model_df = pd.get_dummies(
    model_df,
    drop_first=True
)

print("\nEncoded Dataset Shape:")
print(model_df.shape)

print(
    "\nCategorical Features Converted "
    "Using One-Hot Encoding."
)

# =====================================================
# FEATURES & TARGET
# =====================================================

X = model_df.drop(
    "Selling_Price",
    axis=1
)

y = model_df["Selling_Price"]

print("\nFeature Matrix Shape:")
print(X.shape)

print(
    f"Target Variable Size: {len(y)}"
)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTRAIN TEST SPLIT")
print("-" * 80)

print(
    f"Training Samples : "
    f"{len(X_train)}"
)

print(
    f"Testing Samples  : "
    f"{len(X_test)}"
)

# =====================================================
# MODEL TRAINING
# =====================================================

print("\nTRAINING RANDOM FOREST MODEL")
print("=" * 80)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print(
    "Model Training Completed Successfully."
)

# =====================================================
# PREDICTIONS
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

print("\nMODEL PERFORMANCE")
print("=" * 80)

print(
    f"MAE  : {mae:.3f}"
)

print(
    f"MSE  : {mse:.3f}"
)

print(
    f"RMSE : {rmse:.3f}"
)

print(
    f"R² Score : {r2:.4f}"
)

# =====================================================
# METRIC INTERPRETATION
# =====================================================

print("\nMETRIC INTERPRETATION")
print("-" * 80)

print(
    f"Mean Absolute Error : "
    f"{mae:.3f} Lakhs"
)

print(
    "Average prediction error."
)

print()

print(
    f"Root Mean Squared Error : "
    f"{rmse:.3f} Lakhs"
)

print(
    "Measures prediction accuracy."
)

print()

print(
    f"R² Score : "
    f"{r2:.4f}"
)

if r2 >= 0.95:

    print(
        "Excellent predictive performance."
    )

elif r2 >= 0.90:

    print(
        "Very strong predictive performance."
    )

elif r2 >= 0.80:

    print(
        "Good predictive performance."
    )

else:

    print(
        "Model needs improvement."
    )

# =====================================================
# PREDICTION ERROR ANALYSIS
# =====================================================

errors = abs(
    y_test - y_pred
)

print("\nERROR ANALYSIS")
print("-" * 80)

print(
    f"Average Error : "
    f"{errors.mean():.3f} Lakhs"
)

print(
    f"Maximum Error : "
    f"{errors.max():.3f} Lakhs"
)

print(
    f"Minimum Error : "
    f"{errors.min():.3f} Lakhs"
)

# =====================================================
# MODEL SUMMARY
# =====================================================

print("\nMODEL SUMMARY")
print("=" * 80)

print(
    "Algorithm Used : Random Forest Regressor"
)

print(
    f"Number of Features : "
    f"{X.shape[1]}"
)

print(
    f"Training Samples : "
    f"{len(X_train)}"
)

print(
    f"Testing Samples : "
    f"{len(X_test)}"
)

print(
    f"R² Score : "
    f"{r2:.4f}"
)

print(
    f"Average Prediction Error : "
    f"{mae:.3f} Lakhs"
)

print("=" * 80)




# =====================================================
# GRAPH 7
# ACTUAL VS PREDICTED PRICES
# =====================================================

plt.figure(figsize=(12,7))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.75,
    s=80,
    label="Predictions"
)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    color="red",
    linewidth=2,
    linestyle="--",
    label="Perfect Prediction Line"
)

plt.title(
    "Actual vs Predicted Car Prices",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Actual Selling Price (Lakhs)"
)

plt.ylabel(
    "Predicted Selling Price (Lakhs)"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\nGRAPH 7 INSIGHTS")
print("-" * 80)

prediction_corr = np.corrcoef(
    y_test,
    y_pred
)[0, 1]

print(
    f"Correlation Between Actual and Predicted Prices: "
    f"{prediction_corr:.4f}"
)

print()

if prediction_corr > 0.95:
    print(
        "Predictions closely follow actual values."
    )
elif prediction_corr > 0.85:
    print(
        "Predictions are reasonably accurate."
    )
else:
    print(
        "Prediction quality can be improved."
    )

print()

print(
    "Points closer to the red line indicate "
    "better prediction accuracy."
)

# =====================================================
# GRAPH 8
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

plt.figure(figsize=(12,8))

sns.barplot(
    data=importance_df.head(10),
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 10 Most Important Features",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel("Importance Score")
plt.ylabel("Feature")

plt.tight_layout()
plt.show()

print("\nGRAPH 8 INSIGHTS")
print("-" * 80)

print("Top Features Influencing Price:\n")

for i, row in enumerate(
    importance_df.head(10).itertuples(),
    start=1
):

    print(
        f"{i}. {row.Feature:<25}"
        f"{row.Importance:.4f}"
    )

print()

most_important_feature = (
    importance_df.iloc[0]["Feature"]
)

print(
    f"Most Important Feature: "
    f"{most_important_feature}"
)

print()

print(
    "This feature contributes the most "
    "to the model's predictions."
)

# =====================================================
# SAMPLE PREDICTION
# =====================================================

print("\nSAMPLE PREDICTION")
print("=" * 80)

sample_car = X.iloc[[0]]

actual_price = y.iloc[0]

predicted_price = model.predict(
    sample_car
)[0]

print(
    f"Actual Price    : "
    f"{actual_price:.2f} Lakhs"
)

print(
    f"Predicted Price : "
    f"{predicted_price:.2f} Lakhs"
)

difference = abs(
    actual_price -
    predicted_price
)

print(
    f"Prediction Error: "
    f"{difference:.2f} Lakhs"
)

# =====================================================
# FEATURE ANALYSIS REPORT
# =====================================================

print("\nFEATURE ANALYSIS REPORT")
print("=" * 80)

print(
    f"Most Influential Feature : "
    f"{most_important_feature}"
)

print()

print(
    "Top factors affecting car price:"
)

for feature in importance_df.head(5)["Feature"]:

    print(
        f"• {feature}"
    )

# =====================================================
# CAR VALUATION INSIGHTS
# =====================================================

print("\nCAR VALUATION INSIGHTS")
print("=" * 80)

avg_driven = df["Driven_kms"].mean()

print(
    f"Average Kilometers Driven : "
    f"{avg_driven:,.0f}"
)

print()

print(
    f"Average Vehicle Age : "
    f"{avg_age:.2f} Years"
)

print()

print(
    f"Average Depreciation : "
    f"{depreciation:.2f}%"
)

print()

print(
    "Vehicle age and present market value "
    "are major determinants of selling price."
)

# =====================================================
# BUSINESS RECOMMENDATIONS
# =====================================================

print("\nBUSINESS RECOMMENDATIONS")
print("=" * 80)

print(
    "1. Maintain vehicles properly to reduce "
    "depreciation."
)

print()

print(
    "2. Lower mileage generally results in "
    "higher resale value."
)

print()

print(
    "3. Newer vehicles command significantly "
    "better prices."
)

print()

print(
    "4. Dealers can use ML models to estimate "
    "fair market value."
)

print()

print(
    "5. Buyers can compare predicted prices "
    "with listed prices to identify bargains."
)

# =====================================================
# FINAL PROJECT SUMMARY
# =====================================================

print("\nFINAL PROJECT SUMMARY")
print("=" * 80)

print(
    f"Dataset Size              : {len(df)} Cars"
)

print(
    f"Features Used             : {X.shape[1]}"
)

print(
    f"Average Selling Price     : "
    f"{avg_price:.2f} Lakhs"
)

print(
    f"Average Vehicle Age       : "
    f"{avg_age:.2f} Years"
)

print(
    f"Model Used                : "
    f"Random Forest Regressor"
)

print(
    f"R² Score                  : "
    f"{r2:.4f}"
)

print(
    f"Mean Absolute Error       : "
    f"{mae:.3f} Lakhs"
)

print(
    f"Most Important Feature    : "
    f"{most_important_feature}"
)

print()

if r2 >= 0.95:

    print(
        "Conclusion: Excellent predictive model."
    )

elif r2 >= 0.90:

    print(
        "Conclusion: Very strong predictive model."
    )

elif r2 >= 0.80:

    print(
        "Conclusion: Good predictive model."
    )

else:

    print(
        "Conclusion: Model requires improvement."
    )

print()

print(
    "This project demonstrates how machine "
    "learning can accurately estimate used "
    "car prices using historical vehicle data."
)

print()

print("=" * 80)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 80)

