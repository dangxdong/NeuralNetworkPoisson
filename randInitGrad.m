function W = randInitGrad(l1, l2)

% randomly initializes the weights of a layer with l1 incoming
% and l2 outgoing. 
epsilon = sqrt(6)/sqrt(l2+l1); 
W = rand(l2, 1 + l1) * 2 * epsilon - epsilon;

end
