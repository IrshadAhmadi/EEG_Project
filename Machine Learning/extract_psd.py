import mne  # Import the MNE library for EEG data manipulation
import numpy as np  # Import NumPy for numerical operations
import pandas as pd  # Import pandas for data manipulation
from scipy.signal import welch  # Import Welch's method from SciPy for PSD calculation
import os  # Import os module for file and directory operations

# Function to segment data into fixed-length epochs
"""
 Segment the EEG data into fixed-length epochs.
    Parameters: eeg_data (mne.io.Raw): The raw EEG data.
    epoch_samples (int): The number of samples per epoch.
    Returns: np.ndarray: A 3D array with dimensions (channels, epochs, samples per epoch).
"""
def segment_data(eeg_data, epoch_samples):
    
    # Get the total number of samples in the EEG data
    total_samples = eeg_data.n_times
    # Calculate the number of complete epochs that can be formed
    n_epochs = total_samples // epoch_samples
    # Segment the data into epochs
    segmented_data = eeg_data.get_data()[:, :n_epochs * epoch_samples].reshape(
        (eeg_data.info['nchan'], n_epochs, epoch_samples))
    return segmented_data

# Function to extract features using Welch's method from segmented epochs
"""
    Extract features using Welch's method from segmented epochs.
    Parameters: segmented_data (np.ndarray): Segmented EEG data with dimensions (channels, epochs, samples).
    sfreq (float): Sampling frequency of the EEG data.
    Returns:np.ndarray: A 2D array of features with dimensions (epochs, channels).
"""
def extract_features_from_segmented_welch(segmented_data, sfreq):

    # Get the shape of the segmented data
    n_channels, n_epochs, n_samples = segmented_data.shape
    # Initialize an array to hold the features
    features = np.zeros((n_epochs, n_channels))
    
    # Iterate over each epoch and compute PSD for each epoch using Welch's method
    for epoch_idx in range(n_epochs):
        epoch_data = segmented_data[:, epoch_idx, :]
        for ch_idx in range(n_channels):
            # Calculate the Power Spectral Density (PSD) using Welch's method
            freqs, psd = welch(epoch_data[ch_idx, :], fs=sfreq, nperseg=min(256, n_samples))
            # Take the mean of the PSD across frequency bins of interest (1-50 Hz)
            features[epoch_idx, ch_idx] = np.mean(psd[(freqs >= 1) & (freqs <= 50)])
        
    return features

# Directory containing the EEG datasets
dataset_dir = 'Dataset/False'
# Path to the Excel file where features will be saved
excel_file_path = 'Dataset/False/false_eeg_features.xlsx'

# List all .set files in the directory
eeg_files = [f for f in os.listdir(dataset_dir) if f.endswith('.set')]

# Initialize a DataFrame to hold all features
all_features_df = pd.DataFrame()

# Process each EEG dataset file
for eeg_file in eeg_files:
    # Construct the full file path
    file_path = os.path.join(dataset_dir, eeg_file)
    
    # Load the EEG dataset using MNE
    eeg_data = mne.io.read_raw_eeglab(file_path, preload=True)
    
    # Define the epoch length in seconds
    epoch_length_seconds = 1

    # Calculate the number of samples per epoch
    sfreq = eeg_data.info['sfreq']  # Sampling frequency
    epoch_samples = int(epoch_length_seconds * sfreq)

    # Segment the dataset into fixed-length epochs
    segmented_data = segment_data(eeg_data, epoch_samples)

    # Extract features from segmented data using Welch's method
    features_epoch = extract_features_from_segmented_welch(segmented_data, sfreq)

    # Create a DataFrame for the current file's features
    file_features_df = pd.DataFrame(features_epoch, columns=['F3', 'F4', 'F7', 'F8'])
    file_features_df['File'] = eeg_file  # Add a column for the file name

    # Append the current file's features to the main DataFrame
    all_features_df = pd.concat([all_features_df, file_features_df], ignore_index=True)

# Save the combined features to an Excel file
all_features_df.to_excel(excel_file_path, index=False)

print(f"All features saved to {excel_file_path}")
