import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# =====================================================
# LOAD CLEANED + FEATURE-ENGINEERED DATASET
# =====================================================

df = pd.read_excel("data/World_Power_Dataset_FEATURE_ENGINEERED.xlsx")

# =====================================================
# 1. DISTRIBUTION OF TARGET
# =====================================================

plt.figure(figsize=(8,5))
sns.histplot(df["World_Power_Index"], bins=30, kde=True)
plt.title("Distribution of World Power Index")
plt.xlabel("World Power Index")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# =====================================================
# 2. TOP CORRELATED FEATURES
# =====================================================

corr = df.corr(numeric_only=True)["World_Power_Index"].sort_values(ascending=False)
top_features = corr.index[1:11]

plt.figure(figsize=(9,6))
sns.heatmap(
    df[top_features].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Top Correlated Features with World Power Index")
plt.tight_layout()
plt.show()

# =====================================================
# 3. POLITICAL SYSTEM vs WORLD POWER
# =====================================================

df["Political_System_Label"] = df["Political_System_Type_Hybrid"].apply(
    lambda x: "Hybrid" if x == 1 else "Non-Hybrid"
)

plt.figure(figsize=(6,4))
sns.boxplot(
    x="Political_System_Label",
    y="World_Power_Index",
    data=df,
    order=["Non-Hybrid", "Hybrid"]
)
plt.title("Political System vs World Power Index")
plt.xlabel("Political System Type")
plt.ylabel("World Power Index")
plt.tight_layout()
plt.show()

# =====================================================
# 4. YEAR-WISE TREND
# =====================================================

year_trend = df.groupby("Year")["World_Power_Index"].mean().sort_index()

plt.figure(figsize=(7,4))
year_trend.plot(marker="o")
plt.title("Average World Power Index by Year")
plt.xlabel("Year")
plt.ylabel("Average WPI")
plt.tight_layout()
plt.show()

# =====================================================
# 5. GDP SHARE vs WORLD POWER
# =====================================================

plt.figure(figsize=(7,5))
sns.scatterplot(
    x="Share_of_Global_GDP_pct",
    y="World_Power_Index",
    data=df
)
plt.title("GDP Share vs World Power Index")
plt.xlabel("Share of Global GDP (%)")
plt.ylabel("World Power Index")
plt.tight_layout()
plt.show()

# =====================================================
# 6. POWER INDICES vs WORLD POWER (KEY ADDITION)
# =====================================================

indices = [
    "Economic_Power_Index",
    "Military_Power_Index",
    "Tech_Power_Index",
    "Strategic_Leverage_Index"
]

plt.figure(figsize=(10,7))
for col in indices:
    sns.scatterplot(
        x=df[col],
        y=df["World_Power_Index"],
        label=col
    )

plt.title("Composite Power Indices vs World Power Index")
plt.xlabel("Index Value (Normalized)")
plt.ylabel("World Power Index")
plt.legend()
plt.tight_layout()
plt.show()

# =====================================================
# 7. TOP 10 COUNTRIES BY WORLD POWER INDEX
# =====================================================

top_countries = (
    df.groupby("Country_Name")["World_Power_Index"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(8,5))
top_countries.plot(kind="bar")
plt.title("Top 10 Countries by Average World Power Index")
plt.xlabel("Country")
plt.ylabel("Average World Power Index")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
