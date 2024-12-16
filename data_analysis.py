import pandas as pd
import matplotlib.pyplot as plt
import os

# Comprehensive Healthcare and Claims Analysis
# Sophia Beebe
#
# This script performs analysis on two datasets:
# 1) insurance.csv (US healthcare charges) - Columns: age, sex, bmi, children, smoker, region, charges
#    
#       A. Average charges for smokers vs non-smokers.
#       B. Region with the highest average charges.
#       C. Charges by BMI category.
#
# 2) claim_data.csv (Claims data)
#    Columns: Claim ID, Provider ID, Patient ID, Date of Service, Billed Amount,
#             Procedure Code, Diagnosis Code, Allowed Amount, Paid Amount,
#             Insurance Type, Claim Status, Reason Code, Follow-up Required,
#             AR Status, Outcome
#
#    
#       D. Percentage of denied claims by insurance type (Commercial vs Medicare).
#       E. Correlation between denial (Claim Status == "Denied") and Billed Amount.
#          Also checks proportion of denied claims above and below or equal to $200.
#
# Charts Produced:
# - charges_by_region.png (average charges by region)
# - charges_by_bmi_category.png (average charges by BMI category)
# - denied_claims_by_insurance_type.png (bar chart of denied claims % by insurance type)
#
# Running Instructions:
# 1. Ensure "insurance.csv" and "claim_data.csv" are in the same directory.
# 2. Install dependencies: pip install pandas matplotlib
# 3. Run: python data_analysis.py
#
# Results are printed to the console, and charts are saved as .png files.

def categorize_bmi(bmi):
    """Categorize BMI into groups: Underweight, Normal, Overweight, Obese."""
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def plot_bar_chart(data, title, xlabel, ylabel, output_filename):
    """
    Plot a bar chart given a Pandas Series or dictionary-like object.
    """
    plt.figure(figsize=(8, 5))
    data.plot(kind='bar', color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"A bar chart has been saved as {output_filename}.")

def analyze_insurance_data(df):
    """
    Analyze the insurance dataset:
    A. Average charges for smokers vs non-smokers.
    B. Highest average charges by region.
    C. Charges by BMI category.
    """
    print("=== Insurance Data Analysis ===")

    # A. Average charges for smokers vs non-smokers
    smokers = df[df['smoker'] == 'yes']
    nonsmokers = df[df['smoker'] == 'no']
    avg_charges_smokers = smokers['charges'].mean()
    avg_charges_nonsmokers = nonsmokers['charges'].mean()

    print("A. Average Charges by Smoking Status")
    print(f"   Smokers: ${avg_charges_smokers:,.2f}")
    print(f"   Non-Smokers: ${avg_charges_nonsmokers:,.2f}")
    print()

    # B. Which region has the highest average medical charges?
    charges_by_region = df.groupby('region')['charges'].mean()
    print("B. Average Charges by Region")
    for region, charge in charges_by_region.items():
        print(f"   {region.capitalize()}: ${charge:,.2f}")

    highest_region = charges_by_region.idxmax()
    highest_charges = charges_by_region.max()
    print()
    print(f"   The region with the highest average charges is {highest_region.capitalize()} with ${highest_charges:,.2f}.")
    print()

    # Bar chart for charges by region
    plot_bar_chart(charges_by_region,
                title='Average Healthcare Charges by Region',
                xlabel='Region',
                ylabel='Average Charges ($)',
                output_filename='charges_by_region.png')

    # C. Charges by BMI category
    df['bmi_category'] = df['bmi'].apply(categorize_bmi)
    charges_by_bmi = df.groupby('bmi_category')['charges'].mean().sort_values()
    print("C. Average Charges by BMI Category")
    for category, charge in charges_by_bmi.items():
        print(f"   {category}: ${charge:,.2f}")

    highest_bmi_cat = charges_by_bmi.idxmax()
    highest_bmi_charges = charges_by_bmi.max()
    print()
    print(f"   The BMI category with the highest average charges is {highest_bmi_cat} with ${highest_bmi_charges:,.2f}.")
    print()

    # Bar chart for charges by BMI category
    plot_bar_chart(charges_by_bmi,
                title='Average Healthcare Charges by BMI Category',
                xlabel='BMI Category',
                ylabel='Average Charges ($)',
                output_filename='charges_by_bmi_category.png')

def analyze_claim_data(df_claims):
    """
    Analyze the claims dataset:
    D. Percentage of denied claims by insurance type (Commercial vs Medicare).
    E. Check correlation between denial and billed amount, and examine if claims over $200 are more likely to be denied.
    """
    print("=== Claims Data Analysis ===")

    # Expected columns
    expected_cols = {
        'Claim ID', 'Provider ID', 'Patient ID', 'Date of Service', 'Billed Amount',
        'Procedure Code', 'Diagnosis Code', 'Allowed Amount', 'Paid Amount',
        'Insurance Type', 'Claim Status', 'Reason Code', 'Follow-up Required',
        'AR Status', 'Outcome'
    }
    if not expected_cols.issubset(df_claims.columns):
        print("Warning: 'claim_data.csv' does not have all the expected columns. Adjust the code accordingly.")
    # We'll proceed assuming essential columns are present.

    # Drop any missing values in key columns
    df_claims = df_claims.dropna(subset=['Insurance Type', 'Claim Status', 'Billed Amount'])

    # D. Percentage of denied claims by insurance type (Commercial vs Medicare)
    # Filter for only Commercial and Medicare claims
    df_filtered = df_claims[df_claims['Insurance Type'].isin(['Commercial', 'Medicare'])]
    total_by_type = df_filtered.groupby('Insurance Type')['Claim ID'].count()
    denied_by_type = df_filtered[df_filtered['Claim Status'] == 'Denied'].groupby('Insurance Type')['Claim ID'].count()

    # Calculate percentages
    denied_percentage = (denied_by_type / total_by_type * 100).fillna(0)

    print("D. Percentage of Denied Claims by Insurance Type")
    for itype, pct in denied_percentage.items():
        print(f"   {itype}: {pct:.2f}% denied")

    # Bar chart for denied claims percentage by insurance type
    plot_bar_chart(denied_percentage,
                title='Percentage of Denied Claims by Insurance Type',
                xlabel='Insurance Type',
                ylabel='Percentage Denied (%)',
                output_filename='denied_claims_by_insurance_type.png')

    print()

    # E. Correlation between denial claim status and billed amount
    # Create a binary column: Denied = 1 if Claim Status is 'Denied', else 0
    df_claims['Denied'] = (df_claims['Claim Status'] == 'Denied').astype(int)

    # Compute correlation
    # If correlation is positive, higher billed amounts might be associated with more denials
    # If negative, higher billed amounts might be associated with fewer denials
    # If near zero, little linear relationship.
    correlation = df_claims['Denied'].corr(df_claims['Billed Amount'])

    print("E. Correlation between Denial Status and Billed Amount")
    print(f"   Pearson correlation: {correlation:.4f}")
    print("   (A positive correlation means as Billed Amount increases, denial likelihood tends to increase.)")
    print()

    # Check proportion of denied claims above and below or equal to $200
    high_billed = df_claims['Billed Amount'] > 200
    denied_high = df_claims[high_billed]['Denied'].mean() * 100  # percentage denied when > 200
    denied_low = df_claims[~high_billed]['Denied'].mean() * 100  # percentage denied when <= 200

    print("   Percentage of denied claims when Billed Amount > $200: {:.2f}%".format(denied_high))
    print("   Percentage of denied claims when Billed Amount <= $200: {:.2f}%".format(denied_low))
    print()

def main():
    # Check for datasets
    if not os.path.exists("insurance.csv"):
        print("Error: 'insurance.csv' file not found.")
        return

    if not os.path.exists("claim_data.csv"):
        print("Error: 'claim_data.csv' file not found.")
        return

    # Load insurance data
    try:
        df_insurance = pd.read_csv("insurance.csv")
    except Exception as e:
        print(f"Error loading insurance.csv: {e}")
        return

    # Validate insurance data columns
    expected_insurance_cols = {"age", "sex", "bmi", "children", "smoker", "region", "charges"}
    if not expected_insurance_cols.issubset(df_insurance.columns):
        print("Error: The insurance dataset does not have the expected columns.")
        return

    df_insurance = df_insurance.dropna()

    # Analyze insurance data
    analyze_insurance_data(df_insurance)

    # Load claim data
    try:
        df_claims = pd.read_csv("claim_data.csv")
    except Exception as e:
        print(f"Error loading claim_data.csv: {e}")
        return

    # Analyze claim data
    analyze_claim_data(df_claims)

    print("Analysis complete.")

if __name__ == "__main__":
    main()
