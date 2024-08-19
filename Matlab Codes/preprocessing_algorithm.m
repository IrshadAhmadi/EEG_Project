% You can run this script if you have EEGLAB installed on MATLAB

% full path to the CSV file containing the EEG data
filePath = 'C:\\Users\\Irshad\\Downloads\\eeglab_current\\EEG_project\\Cleaned Dataset\\actual\\subject-13_actual.csv';

% Loads the CSV data into a matrix
% readmatrix reads data from a CSV file into an array, where each column
% corresponds to a time point or an EEG channel.
csvData = readmatrix(filePath);

% Extracts time and EEG data from the loaded matrix
% Assuming the first column is time and the remaining columns are EEG channels
time = csvData(:, 1); % Extract the time column
eegData = csvData(:, 2:end); % Extract EEG data (all columns except the first)

% Defined labels for each EEG channel
% This helps in identifying the channels when visualizing or analyzing the data
channelLabels = {'F8', 'F4', 'Fz', 'F3', 'F7'};

% Initializes EEGLAB
% This step is necessary to set up the environment for processing EEG data
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

% Creates EEG structure in EEGLAB using the imported data
EEG = pop_importdata('setname', 'My EEG Data', ... % Set name for the dataset
                     'data', eegData', ...  % Transpose data to match EEGLAB's format (channels x time)
                     'dataformat', 'array', ... % Specify that the data format is an array
                     'srate', 250, ...  % Define the sampling rate of the EEG data
                     'nbchan', size(eegData, 2), ... % Number of channels in the data
                     'xmin', time(1));  % Specify the starting time value from the data

% Adds channel location information using predefined channel labels
EEG.chanlocs = struct('labels', channelLabels);

% Update the EEGLAB dataset with the imported data
% Stores the EEG data in the global EEGLAB structure
[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, CURRENTSET);

% Refresh EEGLAB interface to reflect the new data
eeglab redraw;

% For Preprocessing Steps

% Remove the channel 'Fz' from the dataset
EEG = pop_select(EEG, 'rmchannel', {'Fz'});

% Load channel scalp location information from an external file
% This includes standard electrode positions to improve analysis accuracy
EEG=pop_chanedit(EEG, {'lookup','C:\\Users\\Irshad\\Downloads\\eeglab_current\\eeglab2024.1\\plugins\\dipfit\\standard_BEM\\elec\\standard_1005.elc'}, ...
    'load',{'C:\\Users\\Irshad\\Downloads\\eeglab_current\\EEG_project\\Electrodes Alocation\\channels_scalp_positions.loc','filetype','autodetect'});

% Checks dataset consistency to ensure data integrity
% This verifies that the dataset is correctly structured and free of errors
EEG = eeg_checkset(EEG);

% Resample the data to 255 Hz for standardization
% Resampling is done to match the desired sampling rate across datasets
EEG = pop_resample(EEG, 255);

% Apply a bandpass filter between 0.5 Hz and 50 Hz to remove noise
% Bandpass filtering helps eliminate low-frequency drift and high-frequency noise
EEG = pop_eegfiltnew(EEG, 'locutoff', 0.5, 'hicutoff', 50, 'plotfreqz', 1);

% Re-reference the data to the average of all channels
% This common referencing method reduces noise and improves data quality
EEG = pop_reref(EEG, []);

% Cleans the data by removing artifacts and noise
% This step removes artifacts based on defined criteria, enhancing signal quality
EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion','off','ChannelCriterion','off','LineNoiseCriterion','off','Highpass','off','BurstCriterion',20,'WindowCriterion',0.25,'BurstRejection','on','Distance','Euclidian','WindowCriterionTolerances',[-Inf 7] );

% Runs Independent Component Analysis (ICA) to identify and remove artifacts
% ICA helps isolate and remove noise/artifacts such as eye blinks
EEG = pop_runica(EEG, 'icatype', 'runica', 'extended',1,'rndreset','yes','interrupt','on','pca',4);

% Plot EEG data post-ICA for visual inspection
% Visual inspection of the cleaned data allows verification of preprocessing steps
pop_eegplot(EEG, 1, 1, 1);

% Save the processed dataset to a file
% The final dataset is saved for further analysis or sharing
EEG = pop_saveset(EEG, 'filename', 'C:\\Users\\Irshad\\Downloads\\eeglab_current\\setfiles\\subject-01_false.set');%, ...
                  %'filepath', 'C:\\Users\\Irshad\\Downloads\\eeglab_current\\setfiles\\false\\');
