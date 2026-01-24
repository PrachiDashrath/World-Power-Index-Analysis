import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("data/World_Power_Dataset_Combined.xlsx")
df = df.sort_values(["Year", "Country_Name"]).reset_index(drop=True)

print(df.columns.tolist())

# 1. The 'Force Multiplier' Check
plt.figure(figsize=(10,6))
sns.boxplot(x='Geostrategic_Chokepoint_Count', y='World_Power_Index', data=df)
plt.title('Impact of Chokepoint Control on World Power')
plt.show()



# 2. Top 10 Countries Trend
top_10_countries = df[df['Year'] == 2024].nlargest(10, 'World_Power_Index')['Country_Name']
trend_df = df[df['Country_Name'].isin(top_10_countries)]

plt.figure(figsize=(12,6))
sns.lineplot(data=trend_df, x='Year', y='World_Power_Index', hue='Country_Name', marker='o')
plt.title('Power Trajectory of Top 10 Nations (2022-2024)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


#3. Correlation 
corr_with_target = df.drop(columns='World_Power_Index')[["Share_of_Global_GDP_pct","Foreign_Exchange_Reserves_USD","Import_Rank_Global","Trade_Partners_Count","Outbound_FDI_USD","Economic_Power_Index"]].corrwith(df['World_Power_Index'])
corr_df = pd.DataFrame(corr_with_target)
plt.figure(figsize=(2, 6)) # Adjust figure size as needed for a single column plot
sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Feature Correlation with Target Variable')
plt.show()


# 3. correlation 
# Dropping non-numeric and target-related columns
numeric_df = df.drop(columns=['World_Power_Index', 'Country_Name', 'Year', 'Reporting_Agency_Code'], errors='ignore')
corr_with_target = numeric_df.corrwith(df['World_Power_Index']).sort_values(ascending=False)

# Convert to DataFrame for heatmap
corr_df = pd.DataFrame(corr_with_target, columns=['Correlation'])

# 2. Define your threshold
threshold = 0.5

# 3. Split the data
high_corr = corr_df[corr_df['Correlation'].abs() >= threshold]
low_corr = corr_df[corr_df['Correlation'].abs() < threshold]

# 4. Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

# Heatmap 1: High Impact
sns.heatmap(high_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', ax=ax1)
ax1.set_title(f'Strong Drivers (abs > {threshold})')

# Heatmap 2: Low Impact / Niche Features
sns.heatmap(low_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', ax=ax2)
ax2.set_title(f'Weak/Niche Features (abs < {threshold})')

plt.tight_layout()
plt.show()