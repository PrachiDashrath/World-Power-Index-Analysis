# =========================================================
# WORLD POWER DATASET
# FINAL DATA CLEANING SCRIPT (ONE INPUT → ONE OUTPUT)
# =========================================================

import pandas as pd
import numpy as np

# =========================================================
# 1. LOAD RAW DATASET
# =========================================================

df = pd.read_excel("World_Power_Dataset.xlsx")
print("Loaded dataset shape:", df.shape)

# =========================================================
# 2. BASIC SANITY CHECKS
# =========================================================

print("\nYear distribution:")
print(df["Year"].value_counts().sort_index())

print("Duplicate rows:", df.duplicated().sum())

# =========================================================
# 3. FIX NUMERIC COLUMNS STORED AS STRINGS
# =========================================================

money_cols = [
    "Foreign_Exchange_Reserves_USD",
    "Defence_Exports_USD"
]

for col in money_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=True)
            .replace("nan", np.nan)
            .astype(float)
        )

# =========================================================
# 4. HANDLE MISSING VALUES
# =========================================================

# ---- Numerical columns → Median imputation
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# ---- Categorical columns → Mode imputation
categorical_cols = [
    "Country_Name",
    "Political_System_Type",
    "International_Aid_Provider",
    "Permanent_UNSC_Membership",
    "Nuclear_Power_Status",
    "Reporting_Agency_Code"
]

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values after cleaning:", df.isnull().sum().sum())

# =========================================================
# 5. CONVERT BINARY YES / NO → 1 / 0
# =========================================================

binary_cols = [
    "International_Aid_Provider",
    "Permanent_UNSC_Membership",
    "Nuclear_Power_Status"
]

for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({"Yes": 1, "No": 0})

# =========================================================
# 6. CLEAN Political_System_Type (STANDARDIZE CATEGORIES)
# =========================================================

def clean_political_system(value):
    if pd.isna(value):
        return "Other"
    value = str(value).lower()
    if "democracy" in value:
        return "Democracy"
    elif "hybrid" in value:
        return "Hybrid"
    else:
        return "Other"

df["Political_System_Type"] = df["Political_System_Type"].apply(clean_political_system)

# =========================================================
# 7. ONE-HOT ENCODE NOMINAL CATEGORICAL FEATURES
# =========================================================

df = pd.get_dummies(
    df,
    columns=["Political_System_Type", "Reporting_Agency_Code"],
    drop_first=True
)

# =========================================================
# 8. SAVE FINAL CLEANED DATASET (ONLY OUTPUT)
# =========================================================

output_file = "World_Power_Dataset_CLEANED.xlsx"
df.to_excel(output_file, index=False)

print("\n DATA CLEANING COMPLETE")
print(f" Final cleaned dataset saved as: {output_file}")
