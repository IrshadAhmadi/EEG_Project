import pandas as pd
from scipy.stats import f_oneway

# Load the Excel files containing the PSD data
# The Excel files should include columns for each frequency band with PSD values
actual_psd = pd.read_excel('PSD/Actual/actual_psd_relative.xlsx')
false_psd = pd.read_excel('PSD/False/false_psd_relative.xlsx')

# Initialize a dictionary to store ANOVA results for each frequency band
results = {}

# List of frequency bands to analyze
bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

# Significance level for determining statistical significance
alpha = 0.05

# Iterate over each frequency band to perform ANOVA
for band in bands:
    # Extract the data for the current band from the actual and false PSD DataFrames
    actual_data = actual_psd[band]
    false_data = false_psd[band]
    
    # Perform one-way ANOVA to compare the actual and false PSD data for the current band
    f_stat, p_value = f_oneway(actual_data, false_data)
    
    # Determine if the difference is statistically significant
    significant = p_value < alpha
    
    # Store the ANOVA results (F-statistic, p-value, and significance) in the results dictionary
    results[band] = {
        'F-statistic': f_stat,
        'p-value': p_value,
        'Significant': 'Yes' if significant else 'No'
    }

# Convert the results dictionary to a DataFrame for better display
anova_results = pd.DataFrame(results).T

# Save the results to an Excel file
anova_results.to_excel('Anova Analysis/ANOVA Results.xlsx', sheet_name='ANOVA Results')

print("ANOVA results have been saved to 'ANOVA Results.xlsx'.")

# Display the ANOVA results
print("ANOVA Results")
anova_results.head()






