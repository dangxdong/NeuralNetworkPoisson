function [J2 grad] = nnCostFuncPoisson_RMSLE(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% Adapted for poisson regression, when the outcome is count number. 
% Adapted to use RMSLE (Root Mean Squared Log Errors, 
% see https://www.kaggle.com/wiki/RootMeanSquaredLogarithmicError)
% as the target to minimise.

% The Gradient function of RMSLE below has been checked, working well. 

% First, reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

% Other functions included:
% sgmdGrad.m

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup variables
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Neural network from input to hidden layer

X1 = [ones(m,1) X]; % so X1 is m * (n+1)
ZZ1 = Theta1 * X1'; % so ZZ1 is n(L2)*m dimensional

AA1 = 1.0 ./ (1.0 + exp(-ZZ1)); % predicted middle layer unit values

% From hidden layer to output layer 
AA1 = [ones(1, m); AA1]; % Now AA1 is(n(L2)+1)*m
ZZ2 = Theta2 * AA1; % ZZ2 is k*m

% Use poisson regression instead of logistic regression for this last lyaer:
AA2 = exp(ZZ2);

% by transposing, we get predicted hh as m * k
hh= AA2'; 


% Calculate cost.
subtraction = log(hh+1) - log(y+1);
J = 1 / m * sum(subtraction.*subtraction);
J = sum(J);  % ensure J is a scalar, not yet regularized
J1 = sqrt(J); % Square-rooted.

% Adding the regularization term
Temp1 = Theta1;
Temp1(:, 1) = 0;
Temp2 = Theta2;
Temp2(:, 1) = 0;
termR = 0.5 * lambda / m * (sum(dot(Temp1,Temp1))+sum(dot(Temp2,Temp2)));

J2 = J1 + termR; % Now cost is regularised.

%%% Implement backpropagation %%%

% To make it easier for vector computation, 
% use the transposes y' and AA2 instead of y and hh, both k*m
% Implementing the gradient function for RMSLE of Poisson regression
delta3 = log(AA2 + 1) - log(y' + 1);
delta3 = delta3 .* AA2 ./ (1 + AA2) ./ J1;

% Then back propagation is the same as plain neural network (logistic regression)
delta2 = Theta2' * delta3;    %(n(L2)+1)*1.
delta2 = delta2(2:end,:);
sigterm2 = sgmdGrad(ZZ1);
delta2 = delta2 .* sigterm2;   

% Give values to the Gradient matrices:
Theta2_grad = Theta2_grad + delta3 * AA1';
Theta1_grad = Theta1_grad + delta2 * X1;

% Do regularization and finalize.
Theta1_grad = 1 / m .* (Theta1_grad + lambda .* Temp1);
Theta2_grad = 1 / m .* (Theta2_grad + lambda .* Temp2);

% Unroll gradients to be returned by the function.
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
