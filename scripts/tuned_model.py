# =====================================================
# World Power Index — Leak-Free Interpretable Model
# ElasticNet + Interaction Features
# STRICT TimeSeriesSplit (Past → Future Only)
# HYPERPARAMETER TUNING (TIME-AWARE)
# STABLE FEATURE IMPORTANCE (Averaged Across Folds)
# =====================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
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
# 1. Load & Sort Data (Temporal Integrity)
# -----------------------------------------------------

df = pd.read_excel("data/World_Power_Dataset_Combined.xlsx")
df = df.sort_values(["Year", "Country_Name"]).reset_index(drop=True)

target = "World_Power_Index"


base_features = [
    "Economic_Power_Index",
    "Military_Power_Index",
    "Tech_Power_Index",
    "Share_of_Global_GDP_pct",
    "Satellite_Ownership_Count",
    "Defense_Expenditure_pct_GDP",
    "Foreign_Exchange_Reserves_USD",
    "Outbound_FDI_USD",
    "Defence_Exports_USD",
    "Permanent_UNSC_Membership",
    "Nuclear_Power_Status",
    "International_Aid_Provider"
]

df[base_features] = df[base_features].apply(pd.to_numeric, errors="coerce")

X = df[base_features].values
# X = df.drop(columns=[target, "Country_Name", "Year"]).values
y = df[target].values
years = df["Year"].values

# -----------------------------------------------------
# 2. Leak-Free Modeling Pipeline
# -----------------------------------------------------

pipeline = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),
    ("scaler", StandardScaler()),
    ("model", ElasticNet(random_state=42))
])

# -----------------------------------------------------
# 3. Hyperparameter Grid (Expanded but Controlled)
# -----------------------------------------------------

param_grid = {
    "poly__degree": [1, 2],
    "poly__interaction_only": [True, False],

    "model__alpha": [0.0005, 0.001, 0.01, 0.1, 1.0],
    "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    "model__fit_intercept": [True, False],
    "model__max_iter": [5000, 10000],
    "model__tol": [1e-4, 1e-3],
}

# -----------------------------------------------------
# 4. STRICT TimeSeriesSplit Hyperparameter Tuning
# -----------------------------------------------------

tscv = TimeSeriesSplit(n_splits=3)

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=tscv,
    scoring="r2",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y)

best_model = grid_search.best_estimator_

print("\nBest Hyperparameters (Leak-Free)")
print(grid_search.best_params_)
print(f"Best CV R²: {grid_search.best_score_:.4f}\n")

# -----------------------------------------------------
# 5. Coefficient Stability Across Time Folds
# -----------------------------------------------------

coef_list = []
ts_r2_scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):

    # Safety check
    assert years[train_idx].max() < years[test_idx].min(), \
        "Temporal leakage detected"

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    best_model.fit(X_train, y_train)
    preds = best_model.predict(X_test)

    ts_r2_scores.append(r2_score(y_test, preds))

    elastic = best_model.named_steps["model"]
    coef_list.append(elastic.coef_)

coef_matrix = np.vstack(coef_list)

# -----------------------------------------------------
# 6. Stable Feature Importance (Mean + Std)
# -----------------------------------------------------

poly = best_model.named_steps["poly"]
# feature_names = poly.get_feature_names_out(df.drop(columns=[target, "Country_Name", "Year"]).columns)
feature_names = poly.get_feature_names_out(base_features)

coef_df = pd.DataFrame(coef_matrix, columns=feature_names)

coef_summary = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Coefficient": coef_df.mean(axis=0),
    "Std_Coefficient": coef_df.std(axis=0),
})

coef_summary["Abs_Mean"] = coef_summary["Mean_Coefficient"].abs()

top_features = coef_summary.sort_values(
    "Abs_Mean", ascending=False
).head(10)

# -----------------------------------------------------
# 7. Final Held-Out Year Evaluation (Last Split Only)
# -----------------------------------------------------

train_idx, test_idx = list(tscv.split(X))[-1]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

# -----------------------------------------------------
# 8. Metrics
# -----------------------------------------------------

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
explained_var = explained_variance_score(y_test, y_pred)

residuals = y_test - y_pred

# -----------------------------------------------------
# 9. Output
# -----------------------------------------------------

print("\nLeak-Free Model Performance Summary (STRICT Temporal)\n")

print("TimeSeriesSplit CV (Past → Future)")
print(f"Mean R² : {np.mean(ts_r2_scores):.4f}")
print(f"Std R²  : {np.std(ts_r2_scores):.4f}\n")

print("Final Held-out Year Performance")
print(f"R² Score           : {r2:.4f}")
print(f"Explained Variance : {explained_var:.4f}")
print(f"RMSE               : {rmse:.4f}")
print(f"MAE                : {mae:.4f}")
print(f"Residual Mean      : {residuals.mean():.6f}")
print(f"Residual Std       : {residuals.std():.4f}\n")

print("STABLE DRIVERS of World Power Index")
print(top_features[[
    "Feature",
    "Mean_Coefficient",
    "Std_Coefficient"
]].to_string(index=False))
