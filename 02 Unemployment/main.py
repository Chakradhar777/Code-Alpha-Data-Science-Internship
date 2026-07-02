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
print("UNEMPLOYMENT ANALYSIS IN INDIA")
print("=" * 80)

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("Unemployment in India.csv")

df.columns = df.columns.str.strip()

print("\nDATASET OVERVIEW")
print("-" * 80)

print("\nFirst 5 Rows:")
print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

# =====================================================
# DATA CLEANING
# =====================================================

df.dropna(inplace=True)

df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True
)

print("\nAfter Cleaning:")
print(df.shape)

# =====================================================
# DESCRIPTIVE STATISTICS
# =====================================================

print("\nSTATISTICAL SUMMARY")
print("-" * 80)

print(df.describe())

print("\nOVERALL INSIGHTS")
print("-" * 80)

avg_unemployment = df[
    "Estimated Unemployment Rate (%)"
].mean()

max_unemployment = df[
    "Estimated Unemployment Rate (%)"
].max()

min_unemployment = df[
    "Estimated Unemployment Rate (%)"
].min()

print(
    f"Average Unemployment Rate : "
    f"{avg_unemployment:.2f}%"
)

print(
    f"Highest Unemployment Rate : "
    f"{max_unemployment:.2f}%"
)

print(
    f"Lowest Unemployment Rate : "
    f"{min_unemployment:.2f}%"
)

# =====================================================
# GRAPH 1
# DISTRIBUTION OF UNEMPLOYMENT
# =====================================================

plt.figure(figsize=(12,6))

sns.histplot(
    data=df,
    x="Estimated Unemployment Rate (%)",
    bins=25,
    kde=True,
    label="Distribution"
)

plt.title(
    "Distribution of Unemployment Rate",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Unemployment Rate (%)",
    fontsize=12
)

plt.ylabel(
    "Frequency",
    fontsize=12
)

plt.legend()

plt.tight_layout()
plt.show()

print("\nGRAPH 1 INSIGHTS")
print("-" * 80)

median_rate = df[
    "Estimated Unemployment Rate (%)"
].median()

print(
    f"Mean Rate   : {avg_unemployment:.2f}%"
)

print(
    f"Median Rate : {median_rate:.2f}%"
)

if avg_unemployment > median_rate:
    print(
        "Distribution is positively skewed."
    )
else:
    print(
        "Distribution is approximately symmetric."
    )

# =====================================================
# GRAPH 2
# TOP 10 STATES
# =====================================================

state_avg = (
    df.groupby("Region")
    ["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(14,7))

sns.barplot(
    x=state_avg.head(10).values,
    y=state_avg.head(10).index
)

plt.title(
    "Top 10 States with Highest Unemployment",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Average Unemployment Rate (%)"
)

plt.ylabel(
    "State"
)

plt.tight_layout()
plt.show()

print("\nGRAPH 2 INSIGHTS")
print("-" * 80)

print(
    f"Highest State : "
    f"{state_avg.idxmax()}"
)

print(
    f"Average Rate  : "
    f"{state_avg.max():.2f}%"
)

print(
    "These states may require stronger "
    "employment-generation policies."
)



# =====================================================
# GRAPH 3
# MONTHLY UNEMPLOYMENT TREND
# =====================================================

monthly_trend = (
    df.groupby(df["Date"].dt.to_period("M"))
    ["Estimated Unemployment Rate (%)"]
    .mean()
)

monthly_trend.index = monthly_trend.index.astype(str)

plt.figure(figsize=(15,7))

plt.plot(
    monthly_trend.index,
    monthly_trend.values,
    marker="o",
    linewidth=2,
    label="Average Unemployment Rate"
)

plt.title(
    "Monthly Unemployment Trend in India",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Month",
    fontsize=12
)

plt.ylabel(
    "Average Unemployment Rate (%)",
    fontsize=12
)

plt.xticks(rotation=90)

plt.grid(True, alpha=0.4)

plt.legend()

plt.tight_layout()
plt.show()

print("\nGRAPH 3 INSIGHTS")
print("-" * 80)

highest_month = monthly_trend.idxmax()
lowest_month = monthly_trend.idxmin()

print(
    f"Highest Unemployment Month : "
    f"{highest_month}"
)

print(
    f"Rate : "
    f"{monthly_trend.max():.2f}%"
)

print()

print(
    f"Lowest Unemployment Month : "
    f"{lowest_month}"
)

print(
    f"Rate : "
    f"{monthly_trend.min():.2f}%"
)

print()

print(
    "The line chart helps identify seasonal "
    "and economic fluctuations in employment."
)

# =====================================================
# GRAPH 4
# RURAL VS URBAN COMPARISON
# =====================================================

plt.figure(figsize=(10,6))

sns.boxplot(
    data=df,
    x="Area",
    y="Estimated Unemployment Rate (%)"
)

plt.title(
    "Rural vs Urban Unemployment Comparison",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Area Type"
)

plt.ylabel(
    "Unemployment Rate (%)"
)

plt.tight_layout()
plt.show()

print("\nGRAPH 4 INSIGHTS")
print("-" * 80)

area_stats = (
    df.groupby("Area")
    ["Estimated Unemployment Rate (%)"]
    .mean()
)

for area, value in area_stats.items():
    print(
        f"{area} Average Rate : "
        f"{value:.2f}%"
    )

print()

highest_area = area_stats.idxmax()

print(
    f"{highest_area} areas experience "
    f"higher unemployment on average."
)

print(
    "Boxplots also reveal variability "
    "and presence of outliers."
)

# =====================================================
# GRAPH 5
# LABOUR PARTICIPATION VS UNEMPLOYMENT
# =====================================================

plt.figure(figsize=(12,7))

sns.scatterplot(
    data=df,
    x="Estimated Labour Participation Rate (%)",
    y="Estimated Unemployment Rate (%)",
    hue="Area",
    s=90
)

plt.title(
    "Labour Participation Rate vs Unemployment",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Labour Participation Rate (%)"
)

plt.ylabel(
    "Unemployment Rate (%)"
)

plt.legend(
    title="Area"
)

plt.tight_layout()
plt.show()

print("\nGRAPH 5 INSIGHTS")
print("-" * 80)

corr_value = (
    df[
        "Estimated Labour Participation Rate (%)"
    ].corr(
        df[
            "Estimated Unemployment Rate (%)"
        ]
    )
)

print(
    f"Correlation Value : "
    f"{corr_value:.3f}"
)

print()

if corr_value > 0.7:
    print(
        "Strong positive relationship detected."
    )

elif corr_value > 0.3:
    print(
        "Moderate positive relationship detected."
    )

elif corr_value > -0.3:
    print(
        "Very weak relationship detected."
    )

else:
    print(
        "Negative relationship detected."
    )

print()

print(
    "This graph helps understand whether "
    "higher labour participation is associated "
    "with changes in unemployment levels."
)

# =====================================================
# INTERMEDIATE REPORT
# =====================================================

print("\nINTERMEDIATE SUMMARY")
print("=" * 80)

print(
    f"Average National Unemployment Rate : "
    f"{avg_unemployment:.2f}%"
)

print(
    f"Highest Unemployment State : "
    f"{state_avg.idxmax()}"
)

print(
    f"Highest Unemployment Month : "
    f"{highest_month}"
)

print(
    f"Labour Participation Correlation : "
    f"{corr_value:.3f}"
)

print("=" * 80)



# =====================================================
# GRAPH 6
# CORRELATION HEATMAP
# =====================================================

numeric_df = df.select_dtypes(include="number")

corr_matrix = numeric_df.corr()

plt.figure(figsize=(10,7))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="RdYlBu_r",
    linewidths=0.5,
    fmt=".2f"
)

plt.title(
    "Correlation Heatmap of Numerical Features",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()
plt.show()

print("\nGRAPH 6 INSIGHTS")
print("-" * 80)

print(
    "Correlation values range from -1 to +1."
)

print(
    "+1  -> Perfect Positive Relationship"
)

print(
    " 0  -> No Relationship"
)

print(
    "-1  -> Perfect Negative Relationship"
)

print()

unemployment_employed_corr = corr_matrix.loc[
    "Estimated Unemployment Rate (%)",
    "Estimated Employed"
]

print(
    f"Correlation between Employment and "
    f"Unemployment: "
    f"{unemployment_employed_corr:.3f}"
)

if unemployment_employed_corr < 0:
    print(
        "As employment increases, "
        "unemployment generally decreases."
    )

print()

print(
    "The heatmap helps identify the most "
    "important relationships in the dataset."
)

# =====================================================
# LOAD COVID DATASET
# =====================================================

print("\n")
print("=" * 80)
print("COVID-19 IMPACT ANALYSIS")
print("=" * 80)

covid_df = pd.read_csv(
    "Unemployment_Rate_upto_11_2020.csv"
)

covid_df.columns = covid_df.columns.str.strip()

covid_df.dropna(inplace=True)

covid_df["Date"] = pd.to_datetime(
    covid_df["Date"],
    dayfirst=True
)

print("\nCovid Dataset Shape:")
print(covid_df.shape)

print("\nColumns:")
print(covid_df.columns.tolist())

print("\nMissing Values:")
print(covid_df.isnull().sum())

# =====================================================
# COVID SUMMARY REPORT
# =====================================================

print("\nCOVID DATASET SUMMARY")
print("-" * 80)

covid_avg = covid_df[
    "Estimated Unemployment Rate (%)"
].mean()

covid_max = covid_df[
    "Estimated Unemployment Rate (%)"
].max()

covid_min = covid_df[
    "Estimated Unemployment Rate (%)"
].min()

print(
    f"Average Unemployment Rate : "
    f"{covid_avg:.2f}%"
)

print(
    f"Maximum Rate : "
    f"{covid_max:.2f}%"
)

print(
    f"Minimum Rate : "
    f"{covid_min:.2f}%"
)

# =====================================================
# GRAPH 7
# COVID IMPACT ANALYSIS
# =====================================================

covid_trend = (
    covid_df.groupby("Date")
    ["Estimated Unemployment Rate (%)"]
    .mean()
)

plt.figure(figsize=(15,7))

plt.plot(
    covid_trend.index,
    covid_trend.values,
    marker="o",
    linewidth=2,
    label="Average Unemployment Rate"
)

plt.axvline(
    pd.Timestamp("2020-03-25"),
    color="red",
    linestyle="--",
    linewidth=2,
    label="National Lockdown"
)

plt.title(
    "Impact of Covid-19 on Unemployment in India",
    fontsize=18,
    fontweight="bold"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Average Unemployment Rate (%)"
)

plt.legend()

plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nGRAPH 7 INSIGHTS")
print("-" * 80)

highest_covid_date = covid_trend.idxmax()

highest_covid_rate = covid_trend.max()

lowest_covid_date = covid_trend.idxmin()

lowest_covid_rate = covid_trend.min()

print(
    f"Highest Covid Unemployment Date : "
    f"{highest_covid_date.date()}"
)

print(
    f"Rate : "
    f"{highest_covid_rate:.2f}%"
)

print()

print(
    f"Lowest Covid Unemployment Date : "
    f"{lowest_covid_date.date()}"
)

print(
    f"Rate : "
    f"{lowest_covid_rate:.2f}%"
)

print()

print(
    "A sharp increase after March 2020 "
    "indicates the economic impact of the "
    "national lockdown."
)

print()

print(
    "Business closures, travel restrictions, "
    "and reduced economic activity contributed "
    "to increased unemployment."
)

print()

print(
    "The gradual decline afterwards suggests "
    "partial economic recovery and reopening."
)

# =====================================================
# COVID ANALYSIS REPORT
# =====================================================

print("\nCOVID IMPACT SUMMARY")
print("=" * 80)

print(
    f"Average Covid Period Unemployment : "
    f"{covid_avg:.2f}%"
)

print(
    f"Peak Covid Unemployment : "
    f"{highest_covid_rate:.2f}%"
)

print(
    f"Peak Occurred On : "
    f"{highest_covid_date.date()}"
)

print()

print(
    "Conclusion:"
)

print(
    "Covid-19 had a substantial impact on "
    "employment across India."
)

print(
    "The unemployment trend clearly shows "
    "a significant rise during lockdown periods."
)

print("=" * 80)



# =====================================================
# GRAPH 8
# REGIONAL UNEMPLOYMENT ANALYSIS
# =====================================================

region_avg = (
    covid_df.groupby("Region")
    ["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(16,10))

sns.barplot(
    x=region_avg.values,
    y=region_avg.index
)

plt.title(
    "Average Unemployment Rate by Region (Covid Period)",
    fontsize=20,
    fontweight="bold"
)

plt.xlabel(
    "Average Unemployment Rate (%)",
    fontsize=12
)

plt.ylabel(
    "Region",
    fontsize=12
)

plt.grid(
    axis="x",
    alpha=0.3
)

plt.tight_layout()
plt.show()

# =====================================================
# GRAPH 8 INSIGHTS
# =====================================================

print("\nGRAPH 8 INSIGHTS")
print("-" * 80)

highest_region = region_avg.idxmax()
highest_region_rate = region_avg.max()

lowest_region = region_avg.idxmin()
lowest_region_rate = region_avg.min()

print(
    f"Highest Unemployment Region : "
    f"{highest_region}"
)

print(
    f"Average Rate : "
    f"{highest_region_rate:.2f}%"
)

print()

print(
    f"Lowest Unemployment Region : "
    f"{lowest_region}"
)

print(
    f"Average Rate : "
    f"{lowest_region_rate:.2f}%"
)

print()

print(
    "Significant variation exists between regions,"
)

print(
    "highlighting differences in economic structure,"
)

print(
    "industrial activity and labour market conditions."
)

# =====================================================
# TOP 5 REGIONS
# =====================================================

print("\nTOP 5 REGIONS WITH HIGHEST UNEMPLOYMENT")
print("=" * 80)

top5 = region_avg.head(5)

for rank, (region, value) in enumerate(
    top5.items(),
    start=1
):
    print(
        f"{rank}. {region:<25} "
        f"{value:.2f}%"
    )

# =====================================================
# BOTTOM 5 REGIONS
# =====================================================

print("\nBOTTOM 5 REGIONS WITH LOWEST UNEMPLOYMENT")
print("=" * 80)

bottom5 = region_avg.tail(5)

for rank, (region, value) in enumerate(
    bottom5.items(),
    start=1
):
    print(
        f"{rank}. {region:<25} "
        f"{value:.2f}%"
    )

# =====================================================
# STATE RANKING TABLE
# =====================================================

print("\nCOMPLETE REGIONAL RANKING")
print("=" * 80)

ranking_df = pd.DataFrame({
    "Region": region_avg.index,
    "Average Unemployment Rate (%)":
    region_avg.values
})

ranking_df["Rank"] = (
    ranking_df[
        "Average Unemployment Rate (%)"
    ]
    .rank(
        ascending=False,
        method="dense"
    )
    .astype(int)
)

ranking_df = ranking_df.sort_values(
    by="Rank"
)

print(
    ranking_df.to_string(index=False)
)

# =====================================================
# COMPARISON REPORT
# =====================================================

print("\nREGIONAL COMPARISON REPORT")
print("=" * 80)

difference = (
    highest_region_rate -
    lowest_region_rate
)

print(
    f"Difference Between Highest and Lowest Region:"
)

print(
    f"{difference:.2f}%"
)

print()

if difference > 10:
    print(
        "Large regional disparity exists."
    )

    print(
        "Employment policies may need "
        "region-specific interventions."
    )

else:
    print(
        "Regional unemployment levels "
        "are relatively consistent."
    )

# =====================================================
# NATIONAL SUMMARY
# =====================================================

national_avg = df[
    "Estimated Unemployment Rate (%)"
].mean()

national_max = df[
    "Estimated Unemployment Rate (%)"
].max()

national_min = df[
    "Estimated Unemployment Rate (%)"
].min()

print("\nNATIONAL EMPLOYMENT SUMMARY")
print("=" * 80)

print(
    f"Average Unemployment Rate : "
    f"{national_avg:.2f}%"
)

print(
    f"Highest Recorded Rate : "
    f"{national_max:.2f}%"
)

print(
    f"Lowest Recorded Rate : "
    f"{national_min:.2f}%"
)

# =====================================================
# KEY FINDINGS
# =====================================================

print("\nKEY FINDINGS")
print("=" * 80)

print(
    "1. Unemployment levels vary considerably "
    "across states and regions."
)

print()

print(
    "2. Urban and Rural areas exhibit different "
    "employment characteristics."
)

print()

print(
    "3. Labour participation rate influences "
    "overall unemployment dynamics."
)

print()

print(
    "4. Covid-19 caused a substantial increase "
    "in unemployment throughout India."
)

print()

print(
    "5. Economic recovery is visible after "
    "the lockdown period."
)

# =====================================================
# RECOMMENDATIONS
# =====================================================

print("\nRECOMMENDATIONS")
print("=" * 80)

print(
    "• Encourage skill development programs."
)

print()

print(
    "• Increase employment opportunities "
    "in high-unemployment regions."
)

print()

print(
    "• Support small and medium enterprises "
    "to generate jobs."
)

print()

print(
    "• Improve labour participation through "
    "education and training initiatives."
)

print()

print(
    "• Strengthen economic resilience during "
    "future crises."
)

# =====================================================
# FINAL EXECUTIVE SUMMARY
# =====================================================

print("\nFINAL EXECUTIVE SUMMARY")
print("=" * 80)

print(
    "This project analyzed unemployment trends "
    "across India using statistical techniques "
    "and data visualization."
)

print()

print(
    "Eight visualizations were used to examine "
    "distribution patterns, regional differences,"
)

print(
    "labour participation, monthly trends, and "
    "the impact of Covid-19."
)

print()

print(
    "The analysis revealed substantial regional "
    "variation and highlighted the significant "
    "economic disruption caused by the pandemic."
)

print()

print(
    "The findings can help policymakers better "
    "understand labour market challenges and "
    "design targeted employment strategies."
)

print()

print("=" * 80)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 80)