function [J2 grad] = nnCostFuncPoisson2hl_RMSLE(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size1, ...
                                   hidden_layer_size2, ...
                                   num_labels, ...
                                   X, y, lambda)
% A 2-hidden-layer ANN model.
% Adapted for poisson regression, when the outcome is count number. 
% Adapted to use RMSLE (Root Mean Squared Log Errors, 
% see https://www.kaggle.com/wiki/RootMeanSquaredLogarithmicError)
% as the target to minimise.

% The Gradient function of RMSLE below has been checked, working well. 

% First, reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

% Other functions included:
% sgmdGrad.m

Theta1 = reshape(nn_params(1:hidden_layer_size1 * (input_layer_size + 1)), ...
                 hidden_layer_size1, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + hidden_layer_size1 * (input_layer_size + 1)):(hidden_layer_size1 * ...
                (input_layer_size + 1)+ hidden_layer_size2 * (hidden_layer_size1 + 1))), ...
                hidden_layer_size2, (hidden_layer_size1 + 1));
                
Theta3 = reshape(nn_params((1 + hidden_layer_size1 * (input_layer_size + 1)+ ...
                hidden_layer_size2 * (hidden_layer_size1 + 1)): end), ...
                num_labels, (hidden_layer_size2 + 1));

% Setup variables
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Theta3_grad = zeros(size(Theta3));

% Neural network from input to hidden layer

X1 = [ones(m,1) X]; % so X1 is m * (n+1)
ZZ1 = Theta1 * X1'; % so ZZ1 is n(L2)*m dimensional

AA1 = 1.0 ./ (1.0 + exp(-ZZ1)); % predicted middle layer unit values

% From hidden layer to output layer 
AA1 = [ones(1, m); AA1]; % Now AA1 is(n(L2)+1)*m
ZZ2 = Theta2 * AA1; % ZZ2 is k*m


AA2 =  1.0 ./ (1.0 + exp(-ZZ2));

AA2 = [ones(1, m); AA2]; % Now AA2 is(n(L3)+1)*m

% As Theta3 is n(L4) * (n(L3)+1) = k*(n(L3)+1)
ZZ3 = Theta3 * AA2; % ZZ3 is n(L4)*m = k*m
AA3 = exp(ZZ3);

hh= AA3';


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
Temp3 = Theta3;
Temp3(:, 1) = 0;
termR = 0.5 * lambda / m * (sum(dot(Temp1,Temp1))+sum(dot(Temp2,Temp2))+sum(dot(Temp3,Temp3)));

J2 = J1 + termR; % Now cost is regularised.

%%% Implement backpropagation %%%

% To make it easier for vector computation, 
% use the transposes y' and AA3 instead of y and hh, both k*m
% Implementing the gradient function for RMSLE of Poisson regression

delta4 = log(AA3 + 1) - log(y' + 1);
delta4 = delta4 .* AA3 ./ (1 + AA3) ./ J1;

delta3 = Theta3' * delta4;
delta3 = delta3(2:end,:);
sigterm3 = sgmdGrad(ZZ2);
delta3 = delta3 .* sigterm3;

% Then back propagation is the same as plain neural network (logistic regression)
delta2 = Theta2' * delta3;    %(n(L2)+1)*1.
delta2 = delta2(2:end,:);
sigterm2 = sgmdGrad(ZZ1);
delta2 = delta2 .* sigterm2;   

% Give values to the Gradient matrices:
Theta3_grad = Theta3_grad + delta4 * AA2';
Theta2_grad = Theta2_grad + delta3 * AA1';
Theta1_grad = Theta1_grad + delta2 * X1;

% Do regularization and finalize.
Theta1_grad = 1 / m .* (Theta1_grad + lambda .* Temp1);
Theta2_grad = 1 / m .* (Theta2_grad + lambda .* Temp2);
Theta3_grad = 1 / m .* (Theta3_grad + lambda .* Temp3);

% Unroll gradients to be returned by the function.
grad = [Theta1_grad(:) ; Theta2_grad(:); Theta3_grad(:)];

end
