% Run your codes like csvread('***.csv', 1, 0) 
% to get X, y, Xval and yval matrices as the training and validation sets.
% Note to keep only numeric values in the data sets.

% Run the model to learn with hidden layer size = 18, lambda = 0.05:
[Theta1, Theta2] = nnPoisson(X, y, 18, 0.05);  

% Predict on the training set
pred1 = predictPoisson(Theta1, Theta2, X);

% Evaluate the prediction performance with RMSLE:
RMSLE1 = calcRMSLE(pred1, y);

% Predict on the validation set
predval1 = predictPoisson(Theta1, Theta2, Xval);

% Evaluate the the prediction performance on the validation set:
RMSLEval1 = calcRMSLE(predval1, yval);

% To view the performance of the prediction, do a simple plot:
plot(predval1, yval, '*');
