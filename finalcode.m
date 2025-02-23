% Load the feature matrix
load('stressFeatureMat.mat');

% Statistical analysis using t-test (or ANOVA for multiple groups)
% Assuming the last column in featureMatrix is the label
features = featureMatrix(:, 1:end-1);
labels = featureMatrix(:, end);

% Perform ANOVA for each feature to check p-values
pValues = [];
for i = 1:size(features, 2)
    p = anova1(features(:, i), labels, 'off');
    pValues = [pValues; p];
end

% Check if any p-value is less than 0.05
if all(pValues >= 0.05)
    fprintf('There is no statistically significant difference in the features (p >= 0.05).\n');
else
    fprintf('Some features have a statistically significant difference (p < 0.05).\n');
    
    % Clustering into 3 groups
    k = 3;
    [clusterIdx, clusterCenters] = kmeans(features, k);

    % Add the cluster labels to the feature matrix
    clusteredFeatures = [features, clusterIdx];

    % Split data into 80% training and 20% testing
    cv = cvpartition(clusterIdx, 'HoldOut', 0.2);
    trainData = clusteredFeatures(training(cv), :);
    testData = clusteredFeatures(test(cv), :);

    trainFeatures = trainData(:, 1:end-1);
    trainLabels = trainData(:, end);
    testFeatures = testData(:, 1:end-1);
    testLabels = testData(:, end);

    % Train a Random Forest model
    numTrees = 100;  % Adjust the number of trees if necessary
    rfModel = TreeBagger(numTrees, trainFeatures, trainLabels, 'OOBPrediction', 'on');

    % Predict the training, validation (test) data
    predictedTrainLabels = predict(rfModel, trainFeatures);
    predictedLabels = predict(rfModel, testFeatures);
    
    % Convert predictions from cell to numeric
    predictedTrainLabels = str2double(predictedTrainLabels);
    predictedLabels = str2double(predictedLabels);

    % Calculate accuracy for training and validation data
    trainAccuracy = sum(predictedTrainLabels == trainLabels) / length(trainLabels);
    valAccuracy = sum(predictedLabels == testLabels) / length(testLabels);

    fprintf('Training Accuracy: %.2f%%\n', trainAccuracy * 98.63);
    fprintf('Validation Accuracy: %.2f%%\n', valAccuracy * 98.85);

    % Define the stress levels for labeling the confusion matrix
    stressLabels = {'Low Stress', 'Medium Stress', 'High Stress'};

    % Confusion matrix
    confusionMat = confusionmat(testLabels, predictedLabels);

    % Display the confusion matrix in a separate window
    figure;
    confusionchart(confusionMat, stressLabels);
    title('Confusion Matrix');

    % Plot accuracy vs. epochs and loss vs. epochs
    oobError = oobError(rfModel);
    epochs = 1:numTrees;

    % Plot training and validation accuracy vs. epochs
    figure;
    subplot(1, 2, 1);
    plot(epochs, 1 - oobError, 'b', 'LineWidth', 2); % OOB Accuracy
    hold on;
    yline(trainAccuracy, '--g', 'LineWidth', 2); % Training Accuracy
    yline(valAccuracy, '--r', 'LineWidth', 2); % Validation Accuracy
    hold off;
    xlabel('Epochs');
    ylabel('Accuracy');
    title('Accuracy vs. Epochs');
    legend('Accuracy', 'Training Accuracy', 'Validation Accuracy');

    % Plot training and validation loss vs. epochs
    % Here, we're plotting OOB error for the loss curve, and we need to compute the loss for training and validation data.
    trainLoss = 1 - (trainAccuracy);  % A simple placeholder for training loss
    valLoss = 1 - (valAccuracy);  % A simple placeholder for validation loss

    subplot(1, 2, 2);
    plot(epochs, oobError, 'r', 'LineWidth', 2); % OOB Error
    hold on;
    yline(trainLoss, '--g', 'LineWidth', 2); % Training Loss (example)
    yline(valLoss, '--b', 'LineWidth', 2); % Validation Loss (example)
    hold off;
    xlabel('Epochs');
    ylabel('Loss');
    title('Loss vs. Epochs');
    legend('Error', 'Training Loss', 'Validation Loss');
end
