# =====================================================
# World Power Index — Leak-Free Interpretable Model
# ElasticNet + Interaction Features
# STRICT TimeSeriesSplit (Past → Future Only)
# =====================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit, GroupKFold
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
# 1. Load & STRICTLY SORT DATA BY YEAR
# -----------------------------------------------------

df = pd.read_excel("data/World_Power_Dataset_Combined.xlsx")

# CRITICAL: ensure temporal order
df = df.sort_values(["Year", "Country_Name"]).reset_index(drop=True)

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
# 2. Leak-Free Pipeline (NO PREPROCESSING LEAKAGE)
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
# 3. STRICT TEMPORAL SPLIT
#    4 Years → 3 Valid Past→Future Splits
# -----------------------------------------------------

tscv = TimeSeriesSplit(n_splits=3)

ts_r2_scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):

    # SAFETY CHECK: training years must be strictly earlier
    assert years[train_idx].max() < years[test_idx].min(), \
        "Temporal leakage detected: future year in training set"

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model_pipeline.fit(X_train, y_train)
    preds = model_pipeline.predict(X_test)

    ts_r2_scores.append(r2_score(y_test, preds))

ts_r2_scores = np.array(ts_r2_scores)

# -----------------------------------------------------
# 4. Country-wise Cross-Validation (ROBUSTNESS CHECK)
# -----------------------------------------------------

gkf = GroupKFold(n_splits=5)
country_r2_scores = []

for train_idx, test_idx in gkf.split(X, y, countries):

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model_pipeline.fit(X_train, y_train)
    preds = model_pipeline.predict(X_test)

    country_r2_scores.append(r2_score(y_test, preds))

country_r2_scores = np.array(country_r2_scores)

# -----------------------------------------------------
# 5. FINAL HELD-OUT YEAR (LAST TEMPORAL SPLIT)
# -----------------------------------------------------

train_idx, test_idx = list(tscv.split(X))[-1]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)

# -----------------------------------------------------
# 6. Metrics
# -----------------------------------------------------

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
explained_var = explained_variance_score(y_test, y_pred)

residuals = y_test - y_pred

# -----------------------------------------------------
# 7. Interpretable Feature Importance
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
).head(7)

# -----------------------------------------------------
# 8. Output
# -----------------------------------------------------

print("\nLeak-Free Model Performance Summary (STRICT Temporal)\n")

print("TimeSeriesSplit (Past → Future Only)")
print(f"Mean R² : {ts_r2_scores.mean():.4f}")
print(f"Std R²  : {ts_r2_scores.std():.4f}\n")

print("Country-wise Robustness CV")
print(f"Mean R² : {country_r2_scores.mean():.4f}")
print(f"Std R²  : {country_r2_scores.std():.4f}\n")

print("Final Held-out Year Performance")
print(f"R² Score            : {r2:.4f}")
print(f"Explained Variance  : {explained_var:.4f}")
print(f"RMSE                : {rmse:.4f}")
print(f"MAE                 : {mae:.4f}")
print(f"Residual Mean       : {residuals.mean():.6f}")
print(f"Residual Std        : {residuals.std():.4f}\n")

print("Top Drivers of World Power Index (Interpretable)")
print(top_features[["Feature", "Coefficient"]].to_string(index=False))
