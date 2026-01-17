import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# =========================================================
# LOAD CLEANED DATASET (FROM data/)
# =========================================================

df = pd.read_excel("data/World_Power_Dataset_CLEANED.xlsx")
scaler = MinMaxScaler()

# =========================================================
# 1. ECONOMIC POWER INDEX
# =========================================================

economic_features = [
    "Share_of_Global_GDP_pct",
    "Foreign_Exchange_Reserves_USD",
    "Outbound_FDI_USD"
]

economic_scaled = scaler.fit_transform(df[economic_features])
df["Economic_Power_Index"] = economic_scaled.mean(axis=1)

# =========================================================
# 2. MILITARY POWER INDEX
# =========================================================

military_features = [
    "Defense_Expenditure_pct_GDP",
    "Defence_Exports_USD",
    "Nuclear_Power_Status",
    "Permanent_UNSC_Membership"
]

military_scaled = scaler.fit_transform(df[military_features])
df["Military_Power_Index"] = military_scaled.mean(axis=1)

# =========================================================
# 3. TECHNOLOGICAL POWER INDEX
# =========================================================

tech_features = [
    "Satellite_Ownership_Count",
    "R_and_D_Expenditure_pct_GDP",
    "Number_of_Data_Centers"
]

tech_scaled = scaler.fit_transform(df[tech_features])
df["Tech_Power_Index"] = tech_scaled.mean(axis=1)

# =========================================================
# 4. STRATEGIC LEVERAGE INDEX
# =========================================================

strategic_features = [
    "Energy_Export_pct",
    "Energy_Import_Dependency_pct",
    "Geostrategic_Chokepoint_Count",
    "Diplomatic_Missions_Count"
]

strategic_scaled = scaler.fit_transform(df[strategic_features])

# Export adds leverage, import dependency reduces it
df["Strategic_Leverage_Index"] = (
    strategic_scaled[:, 0]     # Energy Export
    - strategic_scaled[:, 1]   # Energy Import Dependency
    + strategic_scaled[:, 2]   # Chokepoints
    + strategic_scaled[:, 3]   # Diplomacy
) / 4  # normalize across four components

# =========================================================
# SAVE FEATURE-ENGINEERED DATASET (TO data/)
# =========================================================

output_file = "data/World_Power_Dataset_FEATURE_ENGINEERED.xlsx"
df.to_excel(output_file, index=False)

print(" Feature engineering complete.")
print(f" Saved as: {output_file}")
