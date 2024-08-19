%  To run the script or the program it requires Brainstorm software on MATLAB
%  Open the desired folder based on the subjects IDs
%  
% ---------------------------------------------------------------

% Opens the CSV file containing subject IDs
fileID = fopen('false_ids.csv', 'r');

% Assums the file contains two columns subject IDs and additional info
contents = textscan(fileID, '%s %s', 'Delimiter', ','); % Read the contents of the file using textscan
fclose(fileID); % Closes the file after reading

% Extract the desired column as a list of elements (subject IDs)
files = contents{1}; % Assuming IDs are in the first column

% Display the list of files (subject IDs) to verify correct reading
disp('Subject IDs:');
disp(files);

% Initialize variables for storing input files for processing
sFiles = []; % Stores individual subject files
sFilesGroup = []; % Stores group file for analysis

% List of subject names based on the CSV input
SubjectNames = files;

% Define the path to the EEG data directory
Path = 'False';

% Define the file extension for EEG data files
Format = '.set';

% Initialize RawFiles as an empty cell array to store file paths
RawFiles = {};

% Generate full file paths for each subject's EEG data
for i = 1:length(SubjectNames)
    % Construct full file path for each subject's data
    RawFiles{end+1} = fullfile(Path, [SubjectNames{i} Format]);
end

% Display the generated raw file paths to verify correctness
disp('Raw Files:');
disp(RawFiles);

% Start a new report for the initial data processing
% Reports track the processing steps for later review
bst_report('Start', sFiles);

% Loop through subject names and process each EEG dataset
for i = 1:length(SubjectNames)
    % Process: Import MEG/EEG data with event information
    sFiles{i} = bst_process('CallProcess', 'process_import_data_event', [], [], ...
        'subjectname',  SubjectNames{i}, ... % Name of the subject
        'condition',    '', ...              % Use default condition
        'datafile',     {{RawFiles{i}}, 'EEG-EEGLAB'}, ... % EEG file and format
        'eventname',    'boundary', ...      % Name of the event to process
        'timewindow',   [], ...              % Full time window
        'epochtime',    [-0.1, 0.3], ...     % Epoch time range for analysis
        'split',        0, ...               % Do not split conditions
        'createcond',   1, ...               % Create new conditions for each file
        'ignoreshort',  0, ...               % Include all events, regardless of length
        'channelalign', 1, ...               % Align channels
        'usectfcomp',   1, ...               % Use CTF compensators for MEG/EEG alignment
        'usessp',       1, ...               % Use SSP projectors to remove artifacts
        'freq',         [], ...              % Default frequency range (all)
        'baseline',     [], ...              % Default baseline correction
        'blsensortypes','MEG, EEG');         % Sensor types to process

    % Add the processed subject file to the group file list for averaging
    sFilesGroup = [sFilesGroup, sFiles{i}];
    
    % Process: Average all epochs for each subject file
    sFiles{i} = bst_process('CallProcess', 'process_average', sFiles{i}, [], ...
        'avgtype',       1, ...  % Average all epochs (type 1)
        'avg_func',      1, ...  % Use arithmetic mean for averaging
        'weighted',      0, ...  % Do not use weighted average
        'keepevents',    0);     % Do not retain individual events in the output
end

% Average all epochs for group files and computes the group-level average
sFilesGroup = bst_process('CallProcess', 'process_average', sFilesGroup, [], ...
    'avgtype',       1, ...  % Average all epochs (type 1)
    'avg_func',      1, ...  % Use arithmetic mean
    'weighted',      0, ...  % Do not use weighted average
    'keepevents',    0);     % Do not retain events in the output

% Save and display the report for the initial data processing
ReportFile = bst_report('Save', sFilesGroup);
% Open the saved report for review
bst_report('Open', ReportFile);

% Analysis Part 1: Using averaged data
% --------------------------------------------------

% Specify input files for the first analysis (averaged data)
sFiles1 = {'Group_analysis/actual/data_averaged_240806_1133.mat'};

% Start a new report for the first set of input files
bst_report('Start', sFiles1);

% Computes power spectrum density (PSD) using Welch's method
sFiles1 = bst_process('CallProcess', 'process_psd', sFiles1, [], ...
    'timewindow',  [], ...           % Use full time window
    'win_length',  4, ...            % Window length for PSD computation
    'win_overlap', 50, ...           % 50% overlap between windows
    'units',       'physical', ...   % Units: physical (U^2/Hz)
    'sensortypes', 'EEG', ...        % Process EEG sensor types
    'win_std',     0, ...            % No additional standard deviation applied
    'edit',        struct(...
         'Comment',         'Power,FreqBands', ... % Comment for the process
         'TimeBands',       [], ...                % Time bands not specified
         'Freqs',           {{'delta', '0.5, 4', 'mean'; 'theta', '5, 7', 'mean'; 'alpha', '8, 12', 'mean'; 'beta', '13, 29', 'mean'}}, ... % Frequency bands of interest
         'ClusterFuncTime', 'none', ...            % No time clustering
         'Measure',         'power', ...           % Measure power
         'Output',          'all', ...             % Output all results
         'SaveKernel',      0));                   % Do not save kernel

% Apply spatial smoothing to the first dataset
sFiles1 = bst_process('CallProcess', 'process_ssmooth_surfstat', sFiles1, [], ...
    'fwhm',      3, ...             % Full width at half maximum for smoothing
    'overwrite', 1);                % Overwrite existing data

% Save and display the first analysis report
ReportFile1 = bst_report('Save', sFiles1);
bst_report('Open', ReportFile1);

% Analysis Part 2: Using sLORETA results
% --------------------------------------------------

% Specify input files for the second analysis (sLORETA results)
sFiles2 = {'Group_analysis/boundary/results_sLORETA_EEG_KERNEL_240806_1138.mat'};

% Start a new report for the second set of input files
bst_report('Start', sFiles2);

% Computes power spectrum density (PSD) using Welch's method for sLORETA
sFiles2 = bst_process('CallProcess', 'process_psd', sFiles2, [], ...
    'timewindow',  [], ...           % Use full time window
    'win_length',  4, ...            % Window length for PSD computation
    'win_overlap', 50, ...           % 50% overlap between windows
    'units',       'physical', ...   % Units: physical (U^2/Hz)
    'clusters',    {}, ...           % No clustering specified
    'scoutfunc',   1, ...            % Use mean function for scouts
    'win_std',     0, ...            % No additional standard deviation applied
    'edit',        struct(...
         'Comment',         'Power,FreqBands', ... % Comment for the process
         'TimeBands',       [], ...                % Time bands not specified
         'Freqs',           {{'delta', '0.5, 4', 'mean'; 'theta', '5, 7', 'mean'; 'alpha', '8, 12', 'mean'; 'beta', '13, 29', 'mean'}}, ... % Frequency bands of interest
         'ClusterFuncTime', 'none', ...            % No time clustering
         'Measure',         'power', ...           % Measure power
         'Output',          'all', ...             % Output all results
         'SaveKernel',      0));                   % Do not save kernel

% Applying spatial smoothing to the second dataset
sFiles2 = bst_process('CallProcess', 'process_ssmooth_surfstat', sFiles2, [], ...
    'fwhm',      3, ...             % Full width at half maximum for smoothing
    'overwrite', 1);                % Overwrite existing data

% Spectrum normalization for the second dataset
sFiles2 = bst_process('CallProcess', 'process_tf_norm', sFiles2, [], ...
    'normalize', 'relative2020', ...  % Normalize using relative power (divide by total power)
    'overwrite', 0);                  % Do not overwrite existing data

% Saves and displays the second analysis report
ReportFile2 = bst_report('Save', sFiles2);
bst_report('Open', ReportFile2);
