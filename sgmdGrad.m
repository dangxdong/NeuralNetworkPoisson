function g = sgmdGrad(z)
% returns the gradient of the sigmoid
% evaluated at z
sgmd = 1.0 ./ (1.0 + exp(-z));
g = sgmd .* (1 - sgmd);
end