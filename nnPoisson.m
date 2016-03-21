function [Theta1, Theta2] = nnPoisson(X, y, ...
                           hidden_layer_size = 0, lambda = 0, iteration = 1000, NumTrial = 10)

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

if (hidden_layer_size == 0) 
    hidden_layer_size = min(10, round(input_layer_size*0.5))
end
% Use 3-dimensional matrices to store the candidate theta matrices.
initial_Theta1_Meta = zeros(hidden_layer_size, input_layer_size+1, NumTrial);
initial_Theta2_Meta = zeros(num_labels, hidden_layer_size+1, NumTrial);
initial_nn_params1_Meta = zeros(hidden_layer_size*(input_layer_size+1)+ ...
                                   num_labels*(hidden_layer_size+1), NumTrial);

for ii=1:NumTrial
    initial_Theta1_Meta(:,:,ii) = randInitGrad(input_layer_size, ...
                                                            hidden_layer_size);
    initial_Theta2_Meta(:,:,ii) = randInitGrad(hidden_layer_size, num_labels);
    initial_nn_params1_Meta(:,ii) = [initial_Theta1_Meta(:,:,ii)(:); ...
                                                initial_Theta2_Meta(:,:,ii)(:)];
end

costFunction = @(p) nnCostFuncPoisson_RMSLE(p, ...
                           input_layer_size, ...
                           hidden_layer_size, ...
                           num_labels, X, y, lambda);

options = optimset('MaxIter', iteration);
% Do the modelling
% Strategy: use the first 1000 iterations to run all the initial thetas
% choose the best one to run another 1500 iterations to give the answer.

% Set a reference value to ease the loop.
cost_reference = 100000000.0;
% Run a loop to give the value of nn_params1 with smallest cost1 to nn_paramsM
for jj=1:NumTrial
    initial_nn_params_temp = initial_nn_params1_Meta(:,jj);
    [nn_params1, cost1] = fminunc(costFunction, initial_nn_params_temp, options);
    if cost1(end) < cost_reference
        cost_reference = cost1(end);
        nn_paramsM = nn_params1;
        % Optionally, can set a threshold here to quit the loop early, 
        % once a fairly low cost is reached.
        % if cost_reference < 0.5350
        %     break;
        % end
    end
end
options = optimset('MaxIter', round(iteration*1.5)); 
% can change 1.5 to bigger numbers to achieve smaller cost.

% Pass nn_paramsM to the next round of iterations.
[nn_params, cost] = fminunc(costFunction, nn_paramsM, options);     

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
         hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + hidden_layer_size * (input_layer_size + 1)):end), ...
         num_labels, (hidden_layer_size + 1));

end
