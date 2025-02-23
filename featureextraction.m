% Define the parameters
deltaRange = [0.5, 4];
thetaRange = [4, 8];
alphaRange = [8, 13];
betaRange = [13, 30];
fs = 100;  % Sampling frequency
window = 100 * 30;  % Window size (in samples)

% Path to the directory containing the CSV files
csvDir = './datademo';

% Get a list of all CSV files in the directory
csvFiles = dir(fullfile(csvDir, '*.csv'));

% Initialize the feature matrix
featureMatrix = [];

% Function to extract basic features
function features = basic_feature_extraction(signal)
    features = [mean(signal), var(signal), skewness(signal), kurtosis(signal)];
end

% Loop through each CSV file
for i = 1:length(csvFiles)
    fprintf('[+] Processing file #%d: %s\n', i, csvFiles(i).name);
    
    % Read the current CSV file
    filePath = fullfile(csvDir, csvFiles(i).name);
    data = readtable(filePath);
    
    % Assuming the EEG signal is in the first column of the CSV
    % Adjust this index if the signal is in a different column
    signal = table2array(data(:, 1));
    
    % Remove non-numeric, NaN, and infinite values
    signal = signal(~isnan(signal) & isfinite(signal) & isnumeric(signal));
    
    % Bandpass filter the signal into different frequency bands
    deltaSig = bandpass(signal, deltaRange, fs);
    thetaSig = bandpass(signal, thetaRange, fs);
    alphaSig = bandpass(signal, alphaRange, fs);
    betaSig = bandpass(signal, betaRange, fs);

    % Extract features from the signals in each window
    for n = 1:window:(length(signal) - window + 1)
        % Get the windowed signals
        sigWindow = signal(n:(n + window - 1));
        deltaWindow = deltaSig(n:(n + window - 1));
        thetaWindow = thetaSig(n:(n + window - 1));
        alphaWindow = alphaSig(n:(n + window - 1));
        betaWindow = betaSig(n:(n + window - 1));
        
        % Extract features from each band
        sigFeat = basic_feature_extraction(diff(sigWindow));
        deltaFeat = basic_feature_extraction(diff(deltaWindow));
        thetaFeat = basic_feature_extraction(diff(thetaWindow));
        alphaFeat = basic_feature_extraction(diff(alphaWindow));
        betaFeat = basic_feature_extraction(diff(betaWindow));
        
        % Combine features into one row
        featureRow = [sigFeat, deltaFeat, thetaFeat, alphaFeat, betaFeat];
        
        % Append to the feature matrix
        featureMatrix = [featureMatrix; featureRow];
    end
end

% Save the feature matrix to a MAT file
save('stressFeatureMat.mat', 'featureMatrix');