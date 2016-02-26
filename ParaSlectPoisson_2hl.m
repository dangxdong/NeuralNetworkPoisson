function [opti_hidden_layer_size, opti_lambda, RMSLEvalMatrix, ...
        RMSLEtrainMatrix] = ParaSlectPoisson_2hl(X, y, Xval, yval, ...
        layer_size_list, lambda_list, miniIter=1000)

% Only looking at a single 'best' size or lambda value may not be perfect, so
% it's better to also return the accuracy matrix, to manualy see 
% how the accuracy changes.

% And because of stochastic factors for each different layer size, 
% Run five sets of initialized thetas and select one set with minimal cost
% for each lambda + layer_size setting.

% Will use miniIter as the initial iteration number for theta selection
% and run another 1.5*miniIter to get the final theta values.
% So a complete iteration is 2.5*miniIter times!!
% And actually running 6.5*miniIter times!!!

input_layer_size  = size(X,2);  % 12 original variables
num_labels = size(y,2); % The output y has 4 columns

% set candidate list of of hidden layer sizes and lambda values
if (~exist("layer_size_list")) 
    layer_size_list = [min(10, round(input_layer_size*0.5)), ...
                                        min(20, round(input_layer_size*1.5))];
end

if (~exist("lambda_list"))
    lambda_list = [0.01, 0.1];
end

nn = length(layer_size_list);
nl = length(lambda_list);

% prepare the result matrix, use one extra row 
% to avoid misbehaviour when only one row
RMSLEvalMatrix = zeros(nn+1, nl);
RMSLEtrainMatrix = zeros(nn+1, nl);

for i=1:nn
    % same settings in each outer loop before the inner loop
    hidden_layer_size1 = layer_size_list(i);
    hidden_layer_size2 = layer_size_list(i);
    
    initial_Theta1 = randInitGrad(input_layer_size, hidden_layer_size1);
    initial_Theta2 = randInitGrad(hidden_layer_size1, hidden_layer_size2);
    initial_Theta3 = randInitGrad(hidden_layer_size2, num_labels);
    initial_nn_params1 = [initial_Theta1(:) ; initial_Theta2(:); initial_Theta3(:)];
    
    initial_Theta4 = randInitGrad(input_layer_size, hidden_layer_size1);
    initial_Theta5 = randInitGrad(hidden_layer_size1, hidden_layer_size2);
    initial_Theta6 = randInitGrad(hidden_layer_size2, num_labels);
    initial_nn_params2 = [initial_Theta4(:) ; initial_Theta5(:); initial_Theta6(:)];
    
    initial_Theta7 = randInitGrad(input_layer_size, hidden_layer_size1);
    initial_Theta8 = randInitGrad(hidden_layer_size1, hidden_layer_size2);
    initial_Theta9 = randInitGrad(hidden_layer_size2, num_labels);
    initial_nn_params3 = [initial_Theta7(:) ; initial_Theta8(:); initial_Theta9(:)];
    
    initial_Theta10 = randInitGrad(input_layer_size, hidden_layer_size1);
    initial_Theta11 = randInitGrad(hidden_layer_size1, hidden_layer_size2);
    initial_Theta12 = randInitGrad(hidden_layer_size2, num_labels);
    initial_nn_params4 = [initial_Theta10(:) ; initial_Theta11(:); initial_Theta12(:)];
    
    initial_Theta13 = randInitGrad(input_layer_size, hidden_layer_size1);
    initial_Theta14 = randInitGrad(hidden_layer_size1, hidden_layer_size2);
    initial_Theta15 = randInitGrad(hidden_layer_size2, num_labels);
    initial_nn_params5 = [initial_Theta13(:) ; initial_Theta14(:); initial_Theta15(:)];
    
    for j=1:nl
        % specific settings for each inner loop
        lambda = lambda_list(j);
        costFunction = @(p) nnCostFuncPoisson2hl_RMSLE(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size1, ...
                                   hidden_layer_size2, ...
                                   num_labels, X, y, lambda);
        
        options = optimset('MaxIter', miniIter);
        % Do the modelling
        % Strategy: use the first (miniIter) iterations to run all the initial thetas
        % choose the best one to run another (1.5*miniIter) iterations to give the answer.
        [nn_params1, cost1] = fminunc(costFunction, initial_nn_params1, options);
        [nn_params2, cost2] = fminunc(costFunction, initial_nn_params2, options);
        [nn_params3, cost3] = fminunc(costFunction, initial_nn_params3, options);
        [nn_params4, cost4] = fminunc(costFunction, initial_nn_params4, options);
        [nn_params5, cost5] = fminunc(costFunction, initial_nn_params5, options);
        
        [mincost, maxID] = min([cost1(end), cost2(end), cost3(end), ...
                           cost4(end), cost5(end)]);
        
        switch (maxID)
            case 1
                nn_paramsM = nn_params1;
            case 2
                nn_paramsM = nn_params2;
            case 3
                nn_paramsM = nn_params3;
            case 4
                nn_paramsM = nn_params4;
            otherwise
                nn_paramsM = nn_params5;
        end
        
        options = optimset('MaxIter', round(miniIter*1.5));
        
        % Bring the nn_paramsM to go on optimization
        [nn_params, cost] = fminunc(costFunction, nn_paramsM, options);     

        Theta1 = reshape(nn_params(1:hidden_layer_size1 * (input_layer_size + 1)), ...
                 hidden_layer_size1, (input_layer_size + 1));
        Theta2 = reshape(nn_params((1 + hidden_layer_size1 * (input_layer_size + 1)):(hidden_layer_size1 * ...
                (input_layer_size + 1)+ hidden_layer_size2 * (hidden_layer_size1 + 1))), ...
                hidden_layer_size2, (hidden_layer_size1 + 1));
        Theta3 = reshape(nn_params((1 + hidden_layer_size1 * (input_layer_size + 1)+ ...
                hidden_layer_size2 * (hidden_layer_size1 + 1)):end), ...
                num_labels, (hidden_layer_size2 + 1));
        
        % Predict on the validation set
        [pred, predint] = predictPoisson2hl(Theta1, Theta2, Theta3, X);
        [predval, predintval] = predictPoisson2hl(Theta1, Theta2, Theta3, Xval);
        RMSLE = calcRMSLE(pred, y);
        RMSLEval = calcRMSLE(predval, yval);
        % Do the invert for getting maximum later
        RMSLEtrainMatrix(i,j) = 1./RMSLE;
        RMSLEvalMatrix(i,j) = 1./RMSLEval; 
    end    
end
% Find index for the minimal RMSLE value in the matrix.
% because the values are inverted (1/x), use max() function.
[maxs, rows] = max(RMSLEvalMatrix);
[maxv, colmax] = max(max(RMSLEvalMatrix));
rowmax = rows(colmax);
opti_hidden_layer_size = layer_size_list(rowmax);
opti_lambda = lambda_list(colmax);

% invert back to the real RMSLE values and remove the last row with all zeros.
RMSLEvalMatrix = 1./RMSLEvalMatrix(1:nn,1:nl);
RMSLEtrainMatrix = 1./RMSLEtrainMatrix(1:nn,1:nl);

end
