# =====================================================
# World Power Index Modeling
# Year-wise CV (Primary) and Country-wise CV (Robustness)
# =====================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    explained_variance_score
)

# -----------------------------------------------------
# 1. Load Feature-Engineered Dataset
# -----------------------------------------------------

df = pd.read_excel("data/World_Power_Dataset_FEATURE_ENGINEERED.xlsx")

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

X = df[features]
y = df[target]

years = df["Year"]
countries = df["Country_Name"]

# -----------------------------------------------------
# 2. Feature Scaling
# -----------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------------
# 3. Model Definition
# -----------------------------------------------------

model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=3,
    subsample=0.85,
    random_state=42
)

# -----------------------------------------------------
# 4. Year-wise Cross-Validation (Primary Evaluation)
# -----------------------------------------------------

logo_year = LeaveOneGroupOut()

year_cv_r2 = cross_val_score(
    model,
    X_scaled,
    y,
    cv=logo_year,
    groups=years,
    scoring="r2"
)

# -----------------------------------------------------
# 5. Country-wise Cross-Validation (Robustness Check)
# -----------------------------------------------------

gkf_country = GroupKFold(n_splits=5)

country_cv_r2 = cross_val_score(
    model,
    X_scaled,
    y,
    cv=gkf_country,
    groups=countries,
    scoring="r2"
)

# -----------------------------------------------------
# 6. Final Train-Test Evaluation (Held-out Year)
# -----------------------------------------------------

train_idx, test_idx = list(logo_year.split(X_scaled, y, years))[-1]

X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------------------------------
# 7. Evaluation Metrics
# -----------------------------------------------------

r2_test = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
explained_var = explained_variance_score(y_test, y_pred)

# -----------------------------------------------------
# 8. Feature Importance
# -----------------------------------------------------

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

# -----------------------------------------------------
# 9. Output Results
# -----------------------------------------------------

print("\nModel Performance Summary\n")

print("Year-wise Cross-Validation (Primary)")
print(f"Mean R² : {year_cv_r2.mean():.4f}")
print(f"Std R²  : {year_cv_r2.std():.4f}\n")

print("Country-wise Cross-Validation (Robustness)")
print(f"Mean R² : {country_cv_r2.mean():.4f}")
print(f"Std R²  : {country_cv_r2.std():.4f}\n")

print("Held-out Year Test Performance")
print(f"R² Score            : {r2_test:.4f}")
print(f"Explained Variance  : {explained_var:.4f}")
print(f"RMSE                : {rmse:.4f}")
print(f"MAE                 : {mae:.4f}\n")

print("Top Drivers of World Power Index")
print(importance_df.to_string(index=False))
