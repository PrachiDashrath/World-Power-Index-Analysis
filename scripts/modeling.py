# =====================================================
# World Power Index — Leak-Free Interpretable Model
# ElasticNet + Interaction Features
# Year-wise CV (Primary) + Country-wise CV (Robustness)
# =====================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
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

X = df[base_features].values
y = df[target].values

years = df["Year"].values
countries = df["Country_Name"].values

# -----------------------------------------------------
# 2. Leak-Free Pipeline (KEY FIX)
# -----------------------------------------------------

model_pipeline = Pipeline([
    ("poly", PolynomialFeatures(
        degree=2,
        interaction_only=True,
        include_bias=False
    )),
    ("scaler", StandardScaler()),
    ("model", ElasticNet(
        alpha=0.01,
        l1_ratio=0.5,
        max_iter=5000,
        random_state=42
    ))
])

# -----------------------------------------------------
# 3. Year-wise Cross-Validation (PRIMARY)
# -----------------------------------------------------

logo = LeaveOneGroupOut()
year_r2_scores = []

for train_idx, val_idx in logo.split(X, y, years):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model_pipeline.fit(X_train, y_train)
    preds = model_pipeline.predict(X_val)

    year_r2_scores.append(r2_score(y_val, preds))

year_r2_scores = np.array(year_r2_scores)

# -----------------------------------------------------
# 4. Country-wise Cross-Validation (ROBUSTNESS)
# -----------------------------------------------------

gkf = GroupKFold(n_splits=5)
country_r2_scores = []

for train_idx, val_idx in gkf.split(X, y, countries):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model_pipeline.fit(X_train, y_train)
    preds = model_pipeline.predict(X_val)

    country_r2_scores.append(r2_score(y_val, preds))

country_r2_scores = np.array(country_r2_scores)

# -----------------------------------------------------
# 5. Final Held-out Year Evaluation
# -----------------------------------------------------

train_idx, test_idx = list(logo.split(X, y, years))[-1]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

# -----------------------------------------------------
# 6. Metrics
# -----------------------------------------------------

r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_train.shape[1] - 1)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
explained_var = explained_variance_score(y_test, y_pred)

residuals = y_test - y_pred

# -----------------------------------------------------
# 7. Interpretable Feature Importance (FINAL MODEL)
# -----------------------------------------------------

poly = model_pipeline.named_steps["poly"]
elastic = model_pipeline.named_steps["model"]

feature_names = poly.get_feature_names_out(base_features)
coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": elastic.coef_
})

coef_df["Abs_Coefficient"] = coef_df["Coefficient"].abs()
top_features = coef_df.sort_values(
    "Abs_Coefficient", ascending=False
).head(15)

# -----------------------------------------------------
# 8. Output
# -----------------------------------------------------

print("\nLeak-Free Model Performance Summary\n")

print("Year-wise Cross-Validation (Primary)")
print(f"Mean R² : {year_r2_scores.mean():.4f}")
print(f"Std R²  : {year_r2_scores.std():.4f}\n")

print("Country-wise Cross-Validation (Robustness)")
print(f"Mean R² : {country_r2_scores.mean():.4f}")
print(f"Std R²  : {country_r2_scores.std():.4f}\n")

print("Held-out Year Test Performance")
print(f"R² Score            : {r2:.4f}")
print(f"Adjusted R²         : {adj_r2:.4f}")
print(f"Explained Variance  : {explained_var:.4f}")
print(f"RMSE                : {rmse:.4f}")
print(f"MAE                 : {mae:.4f}")
print(f"Residual Mean       : {residuals.mean():.6f}")
print(f"Residual Std        : {residuals.std():.4f}\n")

print("Top Drivers of World Power Index (Interpretable)")
print(top_features[["Feature", "Coefficient"]].to_string(index=False))
