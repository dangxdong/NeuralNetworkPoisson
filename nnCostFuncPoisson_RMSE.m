function [J2 grad] = nnCostFuncPoisson_RMSE(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% This function is still under develpment. the gradient terms are no right.
% Adapted for poisson regression, when the outcome is count number. 

% First, reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% begin to calculate J and gradients:

X1 = [ones(m,1) X]; % so X1 is m * (n+1)

ZZ1 = Theta1 * X1'; % so ZZ1 is n(L2)*m dimensional
AA1 = sigmoid(ZZ1); % so AA1 is the predicted middle layer unit values

AA1 = [ones(1, m); AA1]; % Now AA1 is(n(L2)+1)*m

% As Theta2 is k * (n(L2)+1)
ZZ2 = Theta2 * AA1; % ZZ2 is k*m = 1*m

% Important!!! here the outcome is counts, rather than classification
% Use poisson regression instead of logistic regression for this last calculation:
AA2 = exp(ZZ2);

hh= AA2'; % by transposing, we get predicted hh as m * k=m*1

% Here the y is already m*1, so just use y to do the calculation
% And we know that hh is the predicted yy, so all the cost is based on hh and yy.

subtraction = hh - y;

J = 1 / m * sum(subtraction.*subtraction);   % now J is also 1*k dimensional, and not yet regularized

J = sum(J);  

J1 = sqrt(J);% not yet regularized

% If using the RMSLE as SQRT, then do :    J1 = sqrt(J); and use J1 for gradient calculation
% And use 

Temp1 = Theta1;
Temp1(:, 1) = 0;
Temp2 = Theta2;
Temp2(:, 1) = 0;
% So Temp1 and Temp2 are modified Theta1 and Theta2, with first columns all zeros
% The regularization term can thus be written as:
termR = 0.5 * lambda / m * (sum(dot(Temp1,Temp1))+sum(dot(Temp2,Temp2)));

J2 = J1 + termR; % Now cost is regularised.

%
% Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. 


% To make it easier to see, y is transposed so each column is a sample record, 
% so we use AA2 instead of hh (remember hh=AA2'), and y' instead of y, both 1*m
delta3 = AA2 - y';

% An extra step here!!!!!! because the last mapping was a Poisson regression,
% the delta3 has to be timed by exp(ZZ2) 
% as the derivative at each ZZ2 value point.  Note exp(ZZ2)=AA2
% And when using RMSLE as the new cost function, the delta3 should be transformed:

% If using the RMSLE as SQRT, then do :
% delta3 = delta3 .* AA2 ./ (1 + AA2) ./ J1;
% If not using the SQRT:
delta3 = delta3 .* AA2 ./ J1;


% use AA2 and AA1 instead of the variable names A3 and A2
% so delta3 is 1*m

% Theta2 is k * (n(L2)+1), in this case is 10*26
% should be 
% delta2 = Theta2' * delta3;   if the last step were linear or logistic regression
delta2 = Theta2' * delta3;
% delta3 is 1*m. The above delta2 is thus (n(L2)+1)*1.

delta2 = delta2(2:end,:);      % we only need delta2 n(L2)*m to clculate Theta1

% use ZZ1 to calculate the sigmoidgradient term; ZZ1 is n(L2)*m dimensional
% instead of only the t-th column of ZZ1
sigterm2 = sigmoidGradient(ZZ1);
    
% so delta2 and sigterm2 are elementwisely multiplicable:
delta2 = delta2 .* sigterm2;   % so we get the delta2 we want    

% Remember that:
% Theta1_grad = zeros(size(Theta1));   n(L2) * n+1
% Theta2_grad = zeros(size(Theta2));   k * (n(L2)+1)

% Use the AA1 values in this t-th column, which is (n(L2)+1) * 1

% as above, delta3 is 1*m, AA1' is m *(n(L2)+1), delta3 * AA1' is 1*(n(L2)+1);
% !!! Note here delta3 * AA1' == delta3(:,1)*AA1'(1,:)+delta3(:,2)*AA1'(2,:)+...
Theta2_grad = Theta2_grad + delta3 * AA1';
% likewise, delta2 n(L2)*m, X1 is m*n+1, they can be multiplied to get Theta1_grad
Theta1_grad = Theta1_grad + delta2 * X1;

% Remember the Temp1 and Temp2 are modified Theta1 and Theta2, with first columns all zeros
% so we can just use them for regularization
Theta1_grad = 1 / m .* (Theta1_grad + lambda .* Temp1);
Theta2_grad = 1 / m .* (Theta2_grad + lambda .* Temp2);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
