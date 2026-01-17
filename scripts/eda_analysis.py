import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# LOAD CLEANED DATASET
# =====================================================

df = pd.read_excel("World_Power_Dataset_CLEANED.xlsx")

# =====================================================
# 1. DISTRIBUTION OF TARGET
# =====================================================

plt.figure(figsize=(8,5))
sns.histplot(df["World_Power_Index"], bins=30, kde=True)
plt.title("Distribution of World Power Index")
plt.xlabel("World Power Index")
plt.ylabel("Count")
plt.show()

# =====================================================
# 2. TOP CORRELATED FEATURES
# =====================================================

corr = df.corr(numeric_only=True)["World_Power_Index"].sort_values(ascending=False)
top_features = corr.index[1:11]

plt.figure(figsize=(8,6))
sns.heatmap(
    df[top_features].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Top Correlated Features with World Power Index")
plt.show()

# =====================================================
# 3. POLITICAL SYSTEM vs WORLD POWER (FIXED PROPERLY)
# =====================================================

# Step 1: force boolean → int
df["Political_System_Type_Hybrid"] = df["Political_System_Type_Hybrid"].astype(int)

# Step 2: create clean categorical label
df["Political_System_Label"] = df["Political_System_Type_Hybrid"].apply(
    lambda x: "Hybrid" if x == 1 else "Non-Hybrid"
)

# Step 3: enforce categorical dtype
df["Political_System_Label"] = df["Political_System_Label"].astype("category")

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
plt.show()

# =====================================================
# 4. YEAR-WISE TREND
# =====================================================

year_trend = df.groupby("Year")["World_Power_Index"].mean()

plt.figure(figsize=(7,4))
year_trend.plot(marker="o")
plt.title("Average World Power Index by Year")
plt.xlabel("Year")
plt.ylabel("Average WPI")
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
plt.show()
