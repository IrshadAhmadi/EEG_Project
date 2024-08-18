import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

# Load the Excel files containing the PSD data
# The Excel files should include columns for each frequency band and a 'Freq' column for frequency labels
actual_psd = pd.read_excel('PSD/Actual/actual_psd_relative.xlsx')
false_psd = pd.read_excel('PSD/False/false_psd_relative.xlsx')

# List of frequency bands to analyze
bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']

# Extract frequency labels for the x-axis from the 'Freq' column
freq_labels = actual_psd['Freq'].tolist()

# Iterate over each frequency band to process and plot the data
for band in bands:
    # Convert the specific band data from DataFrame columns to lists
    actual_data = actual_psd[band].tolist()
    false_data = false_psd[band].tolist()
    
    # Set up the plot with a specific figure size
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot the actual PSD data
    plt.plot(freq_labels, actual_data, marker='o', label='Actual', linestyle='-')
    
    # Plot the false PSD data
    plt.plot(freq_labels, false_data, marker='o', label='False', linestyle='-')
    
    # Customize the plot
    plt.title(f'{band.capitalize()} Band Frequency')  # Set the title of the plot
    plt.xlabel('Frequency Bands')  # Label the x-axis
    plt.ylabel('Relative PSD')  # Label the y-axis
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.legend(loc='upper right')  # Add a legend to distinguish between actual and false data
    plt.grid(True, linestyle='--', alpha=0.5)  # Add a grid with dashed lines and reduced opacity
    
    # Save the figure as an image file in the specified directory
    plt.tight_layout()  # Adjust the layout to prevent clipping of plot elements
    plt.savefig(f'Line Plots/{band}_band_line_plot.png')
    
    # Display the plot
    plt.show()






