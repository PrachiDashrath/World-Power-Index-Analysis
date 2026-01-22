import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler,MinMaxScaler


final_df = pd.DataFrame()
for year in [2021,2022,2023,2024]:
    df = pd.read_excel(f"data/World_Power_Dataset_{year}.xlsx")


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

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

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

    binary_cols = [
        "International_Aid_Provider",
        "Permanent_UNSC_Membership",
        "Nuclear_Power_Status"
    ]

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    # df["Reporting_Agency_Code"].unique()
    df["Reporting_Agency_Code"] = df["Reporting_Agency_Code"].map({
        'SIPRI' : 1,
        'IMF' : 2,
        'UN' : 3,
        'WB' : 4})
    df["Reporting_Agency_Code"] = df["Reporting_Agency_Code"].astype(int)
    

    # df = df.drop(columns=["Reporting_Agency_Code"], errors='ignore',axis=1)

    # df["Political_System_Type"].isna().sum()

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


    df["Political_System_Type"] = df["Political_System_Type"].map({
        'Democracy' : 1,
        'Hybrid' : 2})
    df["Political_System_Type"] = df["Political_System_Type"].astype(int)
    
    '''
    # new approach
    df = pd.get_dummies(
        df,
        columns=["Political_System_Type"],
        prefix="Political_System",
        dtype=int
    )
    '''


    """#Data preprocessing, some data is not in its correct format

    column 1
    """

    df["Defense_Expenditure_pct_GDP"] = df["Defense_Expenditure_pct_GDP"].apply(
        lambda x: x/100 if x > 10 else x
    )

    # Q1 = df["Defence_Exports_USD"].quantile(0.25)
    # Q3 = df["Defence_Exports_USD"].quantile(0.75)
    # IQR = Q3 - Q1

    # lower_bound = Q1 - 1.5 * IQR
    # upper_bound = Q3 + 1.5 * IQR
    # df[(df["Defence_Exports_USD"] >= lower_bound) &
    #             (df["Defence_Exports_USD"] <= upper_bound)]["Defence_Exports_USD"]

    #Invalid as the exports can be higher for more developed countries

    """column 2"""

    # for Satellite_Ownership_Count, the count can be higher than normal behaviour for developed countries

    """column 3"""

    # for R_and_D_Expenditure_pct_GDP, the pct cant go above 10
    df["R_and_D_Expenditure_pct_GDP"] = df["R_and_D_Expenditure_pct_GDP"].apply(
        lambda x: x/100 if x > 10 else x
    )

    """column 4"""

    # for Energy_Import_Dependency_pct, the pct cant go above 100
    df["Energy_Import_Dependency_pct"] = df["Energy_Import_Dependency_pct"].apply(
        lambda x: x/100 if x > 100 else x
    )
    """column 5"""

    # for Energy_Export_pct, the pct cant go above 100
    df["Energy_Export_pct"] = df["Energy_Export_pct"].apply(
        lambda x: x/100 if x > 100 else x
    )


    df_scaled = df.copy()
    df_scaled = df_scaled.drop(["Country_Name","Year"],axis=1)
    scaler = MinMaxScaler()

    # =========================================================
    # 1. ECONOMIC POWER INDEX
    # =========================================================

    economic_features = [
        "Share_of_Global_GDP_pct",
        "Foreign_Exchange_Reserves_USD",
        "Outbound_FDI_USD"
    ]

    economic_scaled = scaler.fit_transform(df_scaled[economic_features])
    df_scaled["Economic_Power_Index"] = economic_scaled.mean(axis=1)

    # =========================================================
    # 2. MILITARY POWER INDEX
    # =========================================================

    military_features = [
        "Defense_Expenditure_pct_GDP",
        "Defence_Exports_USD",
        "Nuclear_Power_Status",
        "Permanent_UNSC_Membership"
    ]

    military_scaled = scaler.fit_transform(df_scaled[military_features])
    df_scaled["Military_Power_Index"] = military_scaled.mean(axis=1)

    # =========================================================
    # 3. TECHNOLOGICAL POWER INDEX
    # =========================================================

    tech_features = [
        "Satellite_Ownership_Count",
        "R_and_D_Expenditure_pct_GDP",
        "Number_of_Data_Centers"
    ]

    tech_scaled = scaler.fit_transform(df_scaled[tech_features])
    df_scaled["Tech_Power_Index"] = tech_scaled.mean(axis=1)

    # =========================================================
    # 4. STRATEGIC LEVERAGE INDEX
    # =========================================================

    strategic_features = [
        "Energy_Export_pct",
        "Energy_Import_Dependency_pct",
        "Geostrategic_Chokepoint_Count",
        "Diplomatic_Missions_Count"
    ]

    strategic_scaled = scaler.fit_transform(df_scaled[strategic_features])

    # Export adds leverage, import dependency reduces it
    df_scaled["Strategic_Leverage_Index"] = (
        strategic_scaled[:, 0]     # Energy Export
        - strategic_scaled[:, 1]   # Energy Import Dependency
        + strategic_scaled[:, 2]   # Chokepoints
        + strategic_scaled[:, 3]   # Diplomacy
    ) / 4  # normalize across four components


    df_scaled[["Country_Name","Year"]] = df[["Country_Name","Year"]]
    final_df = pd.concat([final_df, df_scaled], ignore_index=True)

final_df.to_excel("data/World_Power_Dataset_Combined.xlsx", index=False)