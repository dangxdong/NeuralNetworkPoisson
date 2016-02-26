function [cost] = calcRMSLE(pred, y)
m = size(y, 1);
predlg = log(pred .+ 1);
ylg = log(y .+ 1);
cost = 1 / m * sum((predlg-ylg).*(predlg-ylg));
cost = sum(cost);
cost = sqrt(cost);
end