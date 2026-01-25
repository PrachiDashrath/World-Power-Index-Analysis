# =====================================================
# World Power Index — Focused Descriptive Analysis
# Final Judge-Ready Version (Neat Visuals)
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# -----------------------------------------------------
# 1. Load Dataset
# -----------------------------------------------------

df = pd.read_excel("data/World_Power_Dataset_Combined.xlsx")
df = df.sort_values(["Year", "Country_Name"]).reset_index(drop=True)

target = "World_Power_Index"

features = [
    "Economic_Power_Index",
    "Military_Power_Index",
    "Tech_Power_Index",
    "Strategic_Leverage_Index",
    "Share_of_Global_GDP_pct",
    "Satellite_Ownership_Count",
    "Defense_Expenditure_pct_GDP"
]

df[features + [target]] = df[features + [target]].apply(
    pd.to_numeric, errors="coerce"
)

# -----------------------------------------------------
# 2. Dataset Overview (Console Only)
# -----------------------------------------------------

print("\n================ DATASET OVERVIEW ================\n")
print(f"Total Rows      : {df.shape[0]}")
print(f"Total Countries : {df['Country_Name'].nunique()}")
print(f"Years Covered   : {df['Year'].min()} – {df['Year'].max()}")

print("\nMissing Values:\n")
print(df[features + [target]].isna().sum())


outlier_cols = [
    "Share_of_Global_GDP_pct",
    "Satellite_Ownership_Count",
    "Defense_Expenditure_pct_GDP"
]

df_out = df.copy()

for col in outlier_cols:
    Q1 = df_out[col].quantile(0.25)
    Q3 = df_out[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_out[col] = df_out[col].clip(lower, upper)

# -----------------------------------------------------
# 7. Key Economic Relationship
# -----------------------------------------------------

plt.figure(figsize=(7,5))
sns.scatterplot(
    x=df_out["Share_of_Global_GDP_pct"],
    y=df_out[target],
    alpha=0.6
)
sns.regplot(
    x=df_out["Share_of_Global_GDP_pct"],
    y=df_out[target],
    scatter=False,
    color="red"
)
plt.title("Share of Global GDP vs World Power Index")
plt.xlabel("Share of Global GDP (%)")
plt.ylabel("World Power Index")
plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 8. Temporal Stability — Top 10 Nations
# -----------------------------------------------------

top_10_countries = (
    df[df["Year"] == df["Year"].max()]
    .nlargest(10, "World_Power_Index")["Country_Name"]
)

trend_df = df[df["Country_Name"].isin(top_10_countries)]

plt.figure(figsize=(12,6))
sns.lineplot(
    data=trend_df,
    x="Year",
    y="World_Power_Index",
    hue="Country_Name",
    marker="o"
)
plt.title("Power Trajectory of Top 10 Nations (Recent Years)")
plt.xlabel("Year")
plt.ylabel("World Power Index")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 10. Strong vs Weak Feature Segmentation
# -----------------------------------------------------

numeric_df = df.drop(
    columns=["World_Power_Index", "Country_Name", "Year", "Reporting_Agency_Code"],
    errors="ignore"
)

corr_all = numeric_df.corrwith(df[target]).sort_values(ascending=False)
corr_all_df = pd.DataFrame(corr_all, columns=["Correlation"])

threshold = 0.5

high_corr = corr_all_df[corr_all_df["Correlation"].abs() >= threshold]
low_corr = corr_all_df[corr_all_df["Correlation"].abs() < threshold].head(11)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

sns.heatmap(
    high_corr,
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f",
    ax=ax1
)
ax1.set_title(f"Strong Drivers (|corr| ≥ {threshold})")

sns.heatmap(
    low_corr,
    annot=True,
    cmap="coolwarm",
    center=0,
    fmt=".2f",
    ax=ax2
)
ax2.set_title(f"Weak / Niche Drivers (|corr| < {threshold})")

plt.tight_layout()
plt.show()

# -----------------------------------------------------
# 11. Insight Summary (Console)
# -----------------------------------------------------

print("\n================ KEY DESCRIPTIVE INSIGHTS ================\n")
print("- Economic scale and reserves dominate World Power outcomes.")
print("- Strategic chokepoints act as force multipliers, not linear contributors.")
print("- Top nations show stable power trajectories over time.")
print("- High multicollinearity among dominant drivers justifies ElasticNet.")
print("- Weak drivers add nuance but not predictive dominance.")

# -----------------------------------------------------
# 12. Model Comparison (Year-wise CV R²)
# -----------------------------------------------------

model_results = {
    "ElasticNet (Before Tuning)": 0.7212,
    "ElasticNet (After Tuning)": 0.7708,
    "Gaussian Process": 0.6878,
    "CatBoost": 0.7122,
    "Gradient Boosting": 0.6934
}

model_df = pd.DataFrame(
    list(model_results.items()),
    columns=["Model", "Year-wise CV R²"]
)

plt.figure(figsize=(9,4))

ax = sns.barplot(
    data=model_df,
    x="Model",
    y="Year-wise CV R²"
)

# Add values on top of bars
for p in ax.patches:
    value = p.get_height()
    ax.annotate(
        f"{value:.4f}",
        (p.get_x() + p.get_width() / 2., value),
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        xytext=(0, 3),
        textcoords="offset points"
    )

# Make x-axis labels horizontal & clean
plt.xticks(rotation=0, ha="center")

plt.title("Model Comparison Based on Year-wise Cross-Validated R²")
plt.ylabel("Year-wise CV R²")
plt.ylim(0.6, 0.8)

# Slight bottom padding so labels breathe
plt.tight_layout()
plt.show()
