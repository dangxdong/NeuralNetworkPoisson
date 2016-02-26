function [p, p1]= predictPoisson2hl(Theta1, Theta2, Theta3, X)

% Returns the predicted count of y given the
% trained weights of a neural network (Theta1, Theta2)

% p is in decimal, p1 is rounded integer

% Useful values
m = size(X, 1);
num_labels = size(Theta3, 1);

% input to hidden layer, logistic regression
z1 = [ones(m, 1) X] * Theta1';
h1 = 1.0 ./ (1.0 + exp(-z1));

z2 = [ones(m, 1) h1] * Theta2';
h2 = 1.0 ./ (1.0 + exp(-z2));

% hidden layer to output: Poisson regression
h3 = exp([ones(m, 1) h2] * Theta3');
% just return the h3 values. 
p = h3;
p1 = round(p);

end
