#!/usr/bin/env python3
"""
Customer Service Requests Analysis – NYC 311
Course-end Project 1

This script performs complete data wrangling, exploratory data analysis,
visualizations, response time calculations, and statistical hypothesis testing.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

warnings.filterwarnings('ignore')

# Set visual styles
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.sans-serif'] = 'Helvetica', 'DejaVu Sans', 'Arial'

# Ensure directories exist
os.makedirs("output", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)

DATA_PATH = os.path.join("data", "311_Service_Requests_from_2010_to_Present.csv")

def main():
    print("=" * 80)
    print("NYC 311 CUSTOMER SERVICE REQUESTS ANALYSIS")
    print("=" * 80)

    # 1. LOAD DATASET
    print("\n[Step 1] Loading dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Dataset successfully loaded from: {DATA_PATH}")

    # 2. UNDERSTAND THE DATASET
    print("\n" + "=" * 80)
    print("[Step 2] Understanding the Dataset")
    print("=" * 80)
    print(f"Number of rows    : {df.shape[0]:,}")
    print(f"Number of columns : {df.shape[1]}")
    print(f"Dataset Dimensions: {df.shape}")

    print("\n--- Identifying Variables with Null Values ---")
    missing_values = df.isnull().sum()
    missing_pct = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing_values.index,
        'Missing_Count': missing_values.values,
        'Missing_Percentage': missing_pct.values
    }).sort_values(by='Missing_Percentage', ascending=False).reset_index(drop=True)

    top_missing = missing_df[missing_df['Missing_Count'] > 0]
    print(f"Total columns with missing values: {len(top_missing)} out of {df.shape[1]}")
    print("\nTop 15 columns with highest missing values:")
    print(top_missing.head(15).to_string(index=False))

    # 3. DATA CLEANING & MISSING VALUE TREATMENT
    print("\n" + "=" * 80)
    print("[Step 3] Data Cleaning & Missing Value Treatment")
    print("=" * 80)
    
    # Duplicate records
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate records found: {duplicate_count}")
    if duplicate_count > 0:
        df = df.drop_duplicates()
        print(f"Shape after dropping duplicates: {df.shape}")

    # Missing value treatment for core categorical variables
    print("\nTreating missing values in core categorical attributes...")
    df['Complaint Type'] = df['Complaint Type'].fillna('Unknown')
    df['City'] = df['City'].fillna('Unknown')
    df['Borough'] = df['Borough'].fillna('Unknown')
    # Standardize string capitalization for consistency
    df['City'] = df['City'].str.strip().str.title()
    df['Borough'] = df['Borough'].str.strip().str.title()

    print("Missing count after treatment:")
    print(df[['Complaint Type', 'City', 'Borough']].isnull().sum())

    # Ensure Latitude and Longitude are numeric
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

    # 4. ANALYZE DATE COLUMNS & FILTER INVALID TIMELINES
    print("\n" + "=" * 80)
    print("[Step 4] Analyzing Date Columns & Removing Incorrect Timelines")
    print("=" * 80)
    
    date_columns = ['Created Date', 'Closed Date']
    for col in date_columns:
        print(f"Parsing '{col}' into datetime format...")
        df[col] = pd.to_datetime(df[col], errors='coerce')

    invalid_created = df['Created Date'].isnull().sum()
    invalid_closed = df['Closed Date'].isnull().sum()
    print(f"Invalid / Missing Created Dates: {invalid_created}")
    print(f"Invalid / Missing Closed Dates : {invalid_closed}")

    # Remove records where Created Date is missing
    df = df.dropna(subset=['Created Date'])

    # Remove records where Closed Date < Created Date (incorrect timeline)
    invalid_timeline_mask = (df['Closed Date'].notnull()) & (df['Closed Date'] < df['Created Date'])
    invalid_timeline_count = invalid_timeline_mask.sum()
    print(f"Records with Closed Date earlier than Created Date: {invalid_timeline_count}")

    df = df[~invalid_timeline_mask]
    print(f"Dataset shape after filtering incorrect date timelines: {df.shape}")

    # Feature engineering on date columns
    df['Year'] = df['Created Date'].dt.year
    df['Month'] = df['Created Date'].dt.month
    df['Day'] = df['Created Date'].dt.day
    df['Day_Name'] = df['Created Date'].dt.day_name()
    df['Year_Month'] = df['Created Date'].dt.to_period('M')

    # Monthly complaints trend visualization
    monthly_complaints = df.groupby('Year_Month').size()
    plt.figure(figsize=(14, 6))
    monthly_complaints.plot(kind='line', marker='o', color='#1f77b4', linewidth=2.5, markersize=6)
    plt.title("NYC 311 Monthly Service Requests Trend", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Year - Month", fontsize=12)
    plt.ylabel("Number of Requests", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("visualizations/monthly_complaints.png", dpi=300)
    plt.close()
    print("Saved visualization: visualizations/monthly_complaints.png")

    # 5. FREQUENCY PLOT FOR CITY-WISE COMPLAINTS
    print("\n" + "=" * 80)
    print("[Step 5] City-Wise Complaints Analysis")
    print("=" * 80)
    city_counts = df['City'].value_counts()
    print("Top 15 Cities by Number of Service Requests:")
    print(city_counts.head(15))

    plt.figure(figsize=(14, 7))
    top_cities = city_counts.head(20)
    sns.barplot(x=top_cities.values, y=top_cities.index, palette="mako")
    plt.title("Top 20 Cities by NYC 311 Service Requests", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Number of Requests", fontsize=12)
    plt.ylabel("City", fontsize=12)
    plt.tight_layout()
    plt.savefig("visualizations/city_complaints.png", dpi=300)
    plt.close()
    print("Saved visualization: visualizations/city_complaints.png")

    # 6. SCATTER AND HEXBIN PLOTS FOR BROOKLYN
    print("\n" + "=" * 80)
    print("[Step 6] Complaint Concentration Across Brooklyn")
    print("=" * 80)
    brooklyn_data = df[
        (df['Borough'].str.upper() == 'BROOKLYN') & 
        (df['Latitude'].notnull()) & 
        (df['Longitude'].notnull())
    ].copy()
    print(f"Total geocoded complaints in Brooklyn: {len(brooklyn_data):,}")

    # Scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(brooklyn_data['Longitude'], brooklyn_data['Latitude'], alpha=0.15, s=4, color='#d95f02')
    plt.title("NYC 311 Complaint Concentration in Brooklyn (Scatter Plot)", fontsize=14, fontweight='bold')
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.tight_layout()
    plt.savefig("visualizations/brooklyn_scatter.png", dpi=300)
    plt.close()
    print("Saved visualization: visualizations/brooklyn_scatter.png")

    # Hexbin plot
    plt.figure(figsize=(11, 8.5))
    hb = plt.hexbin(
        brooklyn_data['Longitude'], 
        brooklyn_data['Latitude'], 
        gridsize=50, 
        cmap='YlOrRd', 
        mincnt=1
    )
    cb = plt.colorbar(hb, label='Number of Complaints')
    cb.set_label('Number of Complaints', fontsize=11)
    plt.title("NYC 311 Complaint Density in Brooklyn (Hexbin Plot)", fontsize=14, fontweight='bold')
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.tight_layout()
    plt.savefig("visualizations/brooklyn_hexbin.png", dpi=300)
    plt.close()
    print("Saved visualization: visualizations/brooklyn_hexbin.png")

    # 7. MAJOR TYPES OF COMPLAINTS
    print("\n" + "=" * 80)
    print("[Step 7] Major Types of Complaints")
    print("=" * 80)
    complaint_counts = df['Complaint Type'].value_counts()
    print(f"Total distinct complaint types: {len(complaint_counts)}")
    top_10_complaints = complaint_counts.head(10)
    print("\nTop 10 Types of Complaints:")
    for idx, (c_type, count) in enumerate(top_10_complaints.items(), 1):
        pct = (count / len(df)) * 100
        print(f" {idx:2d}. {c_type:<35}: {count:>7,} ({pct:5.2f}%)")

    # Plot top complaints
    plt.figure(figsize=(12, 7))
    sns.barplot(x=top_10_complaints.values, y=top_10_complaints.index, palette="viridis")
    plt.title("Top 10 NYC 311 Complaint Types", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Number of Complaints", fontsize=12)
    plt.ylabel("Complaint Type", fontsize=12)
    plt.tight_layout()
    plt.savefig("visualizations/top_complaints.png", dpi=300)
    plt.close()
    print("Saved visualization: visualizations/top_complaints.png")

    # Complaint types by city dataset
    print("\nGenerating complaint types by city crosstab dataset...")
    city_complaint_dataset = pd.crosstab(df['City'], df['Complaint Type'])
    output_csv_path = os.path.join("output", "complaint_types_by_city.csv")
    city_complaint_dataset.to_csv(output_csv_path)
    print(f"Saved dataset: {output_csv_path} (Shape: {city_complaint_dataset.shape})")

    # 8. VISUALIZE MAJOR COMPLAINT TYPES IN EACH CITY
    print("\n" + "=" * 80)
    print("[Step 8] Visualizing Major Complaint Types Across Top Cities")
    print("=" * 80)
    top_10_types = top_10_complaints.index.tolist()
    # Filter top 15 cities (excluding 'Unknown' if desired or including top cities)
    top_15_cities = df[df['City'] != 'Unknown']['City'].value_counts().head(15).index.tolist()

    subset_city_comp = df[
        (df['Complaint Type'].isin(top_10_types)) & 
        (df['City'].isin(top_15_cities))
    ]
    city_comp_matrix = pd.crosstab(subset_city_comp['City'], subset_city_comp['Complaint Type'])

    plt.figure(figsize=(14, 8))
    sns.heatmap(city_comp_matrix, cmap='YlGnBu', annot=True, fmt='d', cbar=True, linewidths=0.5)
    plt.title("Top 10 Complaint Types Across Top 15 Cities in NYC", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Complaint Type", fontsize=12)
    plt.ylabel("City", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("visualizations/city_top_complaints_heatmap.png", dpi=300)
    plt.close()
    print("Saved visualization: visualizations/city_top_complaints_heatmap.png")

    # 9. RESPONSE TIME ANALYSIS
    print("\n" + "=" * 80)
    print("[Step 9] Average & Median Response Time Analysis")
    print("=" * 80)
    # Calculate response time in hours
    df['Response_Time_Hours'] = (df['Closed Date'] - df['Created Date']).dt.total_seconds() / 3600.0

    # Filter to closed requests with valid response time
    response_df = df[df['Response_Time_Hours'].notnull() & (df['Response_Time_Hours'] > 0)].copy()
    print(f"Total resolved complaints analyzed for response time: {len(response_df):,}")

    avg_response_by_type = response_df.groupby('Complaint Type')['Response_Time_Hours'].mean().sort_values(ascending=False)
    median_response_by_type = response_df.groupby('Complaint Type')['Response_Time_Hours'].median().sort_values(ascending=False)
    count_by_type = response_df.groupby('Complaint Type')['Response_Time_Hours'].count()

    response_summary = pd.DataFrame({
        'Complaint_Count': count_by_type,
        'Mean_Response_Hours': avg_response_by_type,
        'Median_Response_Hours': median_response_by_type
    }).loc[avg_response_by_type.index]

    print("\nTop 15 Complaint Types with Longest Average Response Time (Hours):")
    print(response_summary.head(15).to_string())

    print("\nResponse Times for Top 10 Most Frequent Complaints:")
    print(response_summary.loc[response_summary.index.isin(top_10_types)].sort_values(by='Complaint_Count', ascending=False).to_string())

    # Visualize average response time for top 20 complaint types by volume
    top_volume_types = response_df['Complaint Type'].value_counts().head(20).index
    top_vol_response = avg_response_by_type.loc[top_volume_types].sort_values(ascending=True)

    plt.figure(figsize=(14, 8))
    bars = plt.barh(top_vol_response.index, top_vol_response.values, color='#4575b4', edgecolor='black', alpha=0.85)
    plt.title("Average Resolution Time (Hours) for Top 20 Most Frequent Complaint Types", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Average Response Time (Hours)", fontsize=12)
    plt.ylabel("Complaint Type", fontsize=12)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, f"{width:.1f}h", ha='left', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig("visualizations/response_time.png", dpi=300)
    plt.close()
    print("Saved visualization: visualizations/response_time.png")

    # 10. STATISTICAL ANALYSIS – CHI-SQUARE TEST OF INDEPENDENCE
    print("\n" + "=" * 80)
    print("[Step 10] Statistical Hypothesis Testing (Chi-Square Test)")
    print("=" * 80)
    print("Hypotheses:")
    print("  H0 (Null Hypothesis)       : Complaint Type and Borough are INDEPENDENT.")
    print("  H1 (Alternative Hypothesis): Complaint Type and Borough are ASSOCIATED / DEPENDENT.")
    print("  Significance Level (alpha) : 0.05\n")

    # Filter to known boroughs and top 10 complaint types for robust test
    valid_boroughs = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island']
    filtered_stat_df = df[
        (df['Complaint Type'].isin(top_10_types)) & 
        (df['Borough'].isin(valid_boroughs))
    ]

    contingency_table = pd.crosstab(
        filtered_stat_df['Borough'], 
        filtered_stat_df['Complaint Type']
    )
    print("Contingency Table (Borough vs Top 10 Complaint Types):")
    print(contingency_table)

    chi2, p_val, dof, expected = chi2_contingency(contingency_table)

    print(f"\nChi-Square Statistic (chi2) : {chi2:,.4f}")
    print(f"P-Value                     : {p_val}")
    print(f"Degrees of Freedom (dof)    : {dof}")
    print(f"Table dimensions            : {contingency_table.shape[0]} boroughs x {contingency_table.shape[1]} complaint types")

    alpha = 0.05
    if p_val < alpha:
        print(f"\nConclusion: Reject the Null Hypothesis (p-value {p_val:.3e} < {alpha}).")
        print("There is highly statistically significant evidence that Complaint Type depends on Borough.")
    else:
        print(f"\nConclusion: Fail to reject the Null Hypothesis (p-value {p_val:.4f} >= {alpha}).")
        print("There is insufficient evidence of an association between Complaint Type and Borough.")

    print("\n" + "=" * 80)
    print("ANALYSIS EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    main()
