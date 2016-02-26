function [Theta1, Theta2] = nnPoisson(X, y, ...
                           hidden_layer_size, lambda = 0, iteration = 1000)

% This function is partly adapted from the homeworks of the online course
% "Machine Learning" on Cousera by Stanford University.

% The optimization function "fmincg.m" from the above course 
% may be used instead of "fminunc.m", which is included in Octave|Matlab.

% Other functions to be included under the same directory as this file:
%
% <randInitGrad.m >(To give initial Theta values for the model.)
%
% <nnCostFuncPoisson_RMSLE.m> (Cost function of a single-hidden-layer ANN model
% specifically adapted for the cases when the outcome Y is count numbers.)
%
% <predictPoisson.m> (for prediction)
%
% Can use a different cost function in place of nnCostFuncPoisson_RMSLE,
% so you can customize the neural network for other distributions of outcome, or
% or other target term to be optimized rather than RMSLE.

% For the input, X is the matrix of predictor variables(columns) in numeric values;
% Y is a single column (multi-column may also be ok) of count numbers as the outcome.

% X should be m rows by n columns, where m is the record number and n is feature number;
% Y shouold be m rows by 1 column, or k columns where there are k different measures 
% or the outcome, should be all count numbers.

% hidden_layer_size should be specified by the user. Usually chosen from
% 0.5 ~ 1.5 times the feature number.

% Lambda is the regularization coefficient. Bigger lambda is prone to 
% prevent overfitting on the the training set, and thus possibly 
% do better when using the thetas on the validation | test sets.

% Theta1 and Theta2 are the output of this function, which can be used 
% by predictPoisson.m, to give predicted values.


input_layer_size  = size(X,2);
num_labels = size(y,2);

initial_Theta1 = randInitGrad(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitGrad(hidden_layer_size, num_labels);
initial_nn_params1 = [initial_Theta1(:) ; initial_Theta2(:)];

initial_Theta3 = randInitGrad(input_layer_size, hidden_layer_size);
initial_Theta4 = randInitGrad(hidden_layer_size, num_labels);
initial_nn_params2 = [initial_Theta3(:) ; initial_Theta4(:)];

initial_Theta5 = randInitGrad(input_layer_size, hidden_layer_size);
initial_Theta6 = randInitGrad(hidden_layer_size, num_labels);
initial_nn_params3 = [initial_Theta5(:) ; initial_Theta6(:)];

initial_Theta7 = randInitGrad(input_layer_size, hidden_layer_size);
initial_Theta8 = randInitGrad(hidden_layer_size, num_labels);
initial_nn_params4 = [initial_Theta7(:) ; initial_Theta8(:)];

initial_Theta9 = randInitGrad(input_layer_size, hidden_layer_size);
initial_Theta10 = randInitGrad(hidden_layer_size, num_labels);
initial_nn_params5 = [initial_Theta9(:) ; initial_Theta10(:)];

costFunction = @(p) nnCostFuncPoisson_RMSLE(p, ...
                           input_layer_size, ...
                           hidden_layer_size, ...
                           num_labels, X, y, lambda);

options = optimset('MaxIter', iteration);
% Do the modelling
% Strategy: use the first 1000 iterations to run all the initial thetas
% choose the best one to run another 1500 iterations to give the answer.
[nn_params1, cost1] = fminunc(costFunction, initial_nn_params1, options);
[nn_params2, cost2] = fminunc(costFunction, initial_nn_params2, options);
[nn_params3, cost3] = fminunc(costFunction, initial_nn_params3, options);
[nn_params4, cost4] = fminunc(costFunction, initial_nn_params4, options);
[nn_params5, cost5] = fminunc(costFunction, initial_nn_params5, options);

[mincost, minID] = min([cost1(end), cost2(end), cost3(end)], ...
                   cost4(end), cost5(end)]);

switch (minID)
    case 1
        nn_paramsM = nn_params1;
    case 2
        nn_paramsM = nn_params2;
    case 3
        nn_paramsM = nn_params3;
    case 4
        nn_paramsM = nn_params4;
    otherwise
        nn_paramsM = nn_params5;
end

options = optimset('MaxIter', round(iteration*1.5));
[nn_params, cost] = fminunc(costFunction, nn_paramsM, options);     

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
         hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + hidden_layer_size * (input_layer_size + 1)):end), ...
         num_labels, (hidden_layer_size + 1));


end