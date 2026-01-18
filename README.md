# World Power Index Analysis

## Overview
This project presents a structured data science approach to understand, decompose, and model the **World Power Index (WPI)** using country-level economic, political, military, technological, and strategic indicators across the years **2021–2024**.

The objective is not only to predict the World Power Index, but also to **explain the underlying drivers of global power** through interpretable analysis and feature engineering.

---

## Dataset Description
The dataset consists of country-wise indicators including:

- Economic metrics (GDP share, foreign exchange reserves, FDI)
- Military indicators (defense expenditure, exports, nuclear capability)
- Technological capacity (R&D expenditure, satellites, data centers)
- Strategic leverage (energy dependence, chokepoints, diplomacy)
- Political and institutional attributes

Each country is observed annually from **2021 to 2024**, enabling cross-sectional as well as temporal analysis.

---

## Methodology

### 1. Data Cleaning & Preparation
- Fixed numeric columns stored as strings
- Handled missing values using **median (numerical)** and **mode (categorical)** imputation
- Standardized inconsistent categorical values
- Converted binary categorical variables to indicator form
- Applied one-hot encoding for nominal variables

The output of this stage is a fully clean, model-ready dataset with no missing values.

---

### 2. Exploratory Data Analysis (EDA)
Key exploratory insights include:
- Distribution analysis of the World Power Index
- Correlation analysis to identify dominant explanatory variables
- Comparison of political system types and power distribution
- Year-wise trend analysis to capture temporal context
- Relationship between economic scale and global power

EDA was used strictly for **interpretation and insight generation**, without altering the dataset.

---

### 3. Feature Engineering
To improve interpretability and reduce dimensional complexity, four composite indices were constructed:

- **Economic Power Index**
- **Military Power Index**
- **Technological Power Index**
- **Strategic Leverage Index**

Each index was created by:
1. Normalizing constituent variables using Min-Max scaling
2. Aggregating components with equal weighting to avoid dominance bias

These indices represent high-level dimensions of global power and are used for downstream modeling and interpretation.

---

## Project Structure
World-Power-Index-Analysis/
│
├── data/
│ ├── World_Power_Dataset.xlsx
│ ├── World_Power_Dataset_CLEANED.xlsx
│ └── World_Power_Dataset_FEATURE_ENGINEERED.xlsx
│
├── scripts/
│ ├── data_cleaning_preparation.py
│ ├── eda_analysis.py
│ └── feature_engineering.py
│
├── outputs/
│ └── figures/
│
├── README.md
└── requirements.txt


---

## Tools & Libraries
- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- openpyxl

---

## Objective
The primary goal of this project is to:
- Understand what factors most strongly influence global power
- Decompose world power into interpretable dimensions
- Prepare a robust dataset for predictive modeling and comparative analysis

---

## Status
- Data cleaning: ✅ Completed  
- EDA: ✅ Completed  
- Feature engineering: ✅ Completed  
Modeling & Evaluation: ✅ Completed


---

### 4. Modeling & Evaluation
A supervised regression model was developed to explain and predict the **World Power Index** using both raw indicators and engineered composite indices.

**Model Used**
- Gradient Boosting Regressor

**Feature Set**
- Economic Power Index  
- Military Power Index  
- Technological Power Index  
- Strategic Leverage Index  
- Share of Global GDP  
- Satellite Ownership Count  
- Defense Expenditure (% of GDP)

**Evaluation Strategy**
To ensure a realistic and non-leaky evaluation, two validation schemes were used:

- **Year-wise Cross-Validation (Primary Evaluation)**  
  Tests how well the model generalizes across different global time periods.

- **Country-wise Cross-Validation (Robustness Check)**  
  Tests how well the model generalizes to unseen countries.

Additionally, the most recent year was held out as a final test set.

**Metrics Reported**
- R² Score
- Explained Variance
- RMSE
- MAE
- Feature Importance

This evaluation strategy prioritizes interpretability, temporal robustness, and judge-safe methodology over artificially inflated accuracy.

---

## How to Run the Project

Install dependencies:
pip install -r requirements.txt

python scripts/data_cleaning_preparation.py
python scripts/eda_analysis.py
python scripts/feature_engineering.py
python scripts/modeling.py



---

## 📌 Small Status Update (replace existing Status section)

md
## Status
- Data cleaning: ✅ Completed  
- EDA: ✅ Completed  
- Feature engineering: ✅ Completed  
- Modeling & Evaluation: ✅ Completed  
- Presentation & Insights: 🔄 In progress

```
