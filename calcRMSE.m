function [cost] = calcRMSE(pred, y)
m = size(y, 1);
cost = 1 / m * sum((pred-y).*(pred-y));
cost = sum(cost);
cost = sqrt(cost);
end