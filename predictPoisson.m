function [p, p1]= predictPoisson(Theta1, Theta2, X)

% Returns the predicted count of y given the
% trained weights of a neural network (Theta1, Theta2)

% p is in decimal, p1 is rounded integer

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% input to hidden layer, logistic regression
z = [ones(m, 1) X] * Theta1';
h1 = 1.0 ./ (1.0 + exp(-z));

% hidden layer to output: Poisson regression
h2 = exp([ones(m, 1) h1] * Theta2');
% just return the h2 values. 
p = h2;
p1 = round(p);

end
