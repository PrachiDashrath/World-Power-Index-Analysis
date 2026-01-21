# =====================================================
# World Power Index Modeling
# ElasticNet with Interpretable Interaction Features
# Year-wise CV (Primary) + Country-wise CV (Robustness)
# =====================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    explained_variance_score
)

# -----------------------------------------------------
# 1. Load Dataset
# -----------------------------------------------------

df = pd.read_excel("data/World_Power_Dataset_Combined.xlsx")

target = "World_Power_Index"

base_features = [
    "Economic_Power_Index",
    "Military_Power_Index",
    "Tech_Power_Index",
    "Strategic_Leverage_Index",
    "Share_of_Global_GDP_pct",
    "Satellite_Ownership_Count",
    "Defense_Expenditure_pct_GDP"
]

df[base_features] = df[base_features].apply(pd.to_numeric, errors="coerce")

X = df[base_features]
y = df[target]

years = df["Year"]
countries = df["Country_Name"]

# -----------------------------------------------------
# 2. Interaction Features (RENAMED FOR CLARITY)
# -----------------------------------------------------

poly = PolynomialFeatures(
    degree=2,
    interaction_only=True,
    include_bias=False
)

X_poly = poly.fit_transform(X)
raw_feature_names = poly.get_feature_names_out(base_features)

# Clean interaction names (A B → A x B)
clean_feature_names = []
for name in raw_feature_names:
    if " " in name:
        parts = name.split(" ")
        clean_feature_names.append(f"{parts[0]} x {parts[1]}")
    else:
        clean_feature_names.append(name)

X_poly = pd.DataFrame(X_poly, columns=clean_feature_names)

# -----------------------------------------------------
# 3. Scaling
# -----------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# -----------------------------------------------------
# 4. ElasticNet Model (Regularized & Interpretable)
# -----------------------------------------------------

model = ElasticNet(
    alpha=0.01,
    l1_ratio=0.5,
    max_iter=5000,
    random_state=42
)

# -----------------------------------------------------
# 5. Year-wise Cross-Validation (PRIMARY)
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
# 6. Country-wise Cross-Validation (ROBUSTNESS)
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
# 7. Held-out Year Evaluation
# -----------------------------------------------------

train_idx, test_idx = list(logo_year.split(X_scaled, y, years))[-1]

X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------------------------------
# 8. Metrics
# -----------------------------------------------------

r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
explained_var = explained_variance_score(y_test, y_pred)

residuals = y_test - y_pred

# -----------------------------------------------------
# 9. Feature Importance (Top Drivers)
# -----------------------------------------------------

importance_df = pd.DataFrame({
    "Feature": clean_feature_names,
    "Coefficient": model.coef_
})

importance_df["Abs_Coefficient"] = importance_df["Coefficient"].abs()
importance_df = importance_df.sort_values(
    "Abs_Coefficient", ascending=False
).head(15)

# -----------------------------------------------------
# 10. Output
# -----------------------------------------------------

print("\nModel Performance Summary\n")

print("Year-wise Cross-Validation (Primary)")
print(f"Mean R² : {year_cv_r2.mean():.4f}")
print(f"Std R²  : {year_cv_r2.std():.4f}\n")

print("Country-wise Cross-Validation (Robustness)")
print(f"Mean R² : {country_cv_r2.mean():.4f}")
print(f"Std R²  : {country_cv_r2.std():.4f}\n")

print("Held-out Year Test Performance")
print(f"R² Score            : {r2:.4f}")
print(f"Adjusted R²         : {adj_r2:.4f}")
print(f"Explained Variance  : {explained_var:.4f}")
print(f"RMSE                : {rmse:.4f}")
print(f"MAE                 : {mae:.4f}")
print(f"Residual Mean       : {residuals.mean():.6f}")
print(f"Residual Std        : {residuals.std():.4f}\n")

print("Top Drivers of World Power Index")
print(importance_df[["Feature", "Coefficient"]].to_string(index=False))
