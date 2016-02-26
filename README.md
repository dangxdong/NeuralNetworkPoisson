# NeuralNetworkPoisson
A Neural Network model in Matlab | Octave for count number prediction.
### First released date: 2016-02-26
### Author: Xiaodong Dang, dangxdong@hotmail.com
### First released date: 2016-02-26

 Implementing artificial neural network 
 with a single hidden layer
 adapted to predict count numbers using Poisson Regression.

 nnCostFuncPoisson_RMSLE.m 
 is the kernel of this model,
 which is adapted from a simple neural network model.
 RMSLE is used as optimization target in the cost function
 see the nnCostFuncPoisson_RMSLE.m
 for more details.

 nnPoisson.m 
 is the entry point, in the form of 
 [Theta1, Theta2] = nnPoisson(X, y, hidden_layer_size, lambda = 0, iteration = 1000)
 where users put in the predictor matrix (all numbers) as X,
 the outcome (in count number) as y (can be more than one column).
 specify the hidden layer size, lambda (as regularization coefficient) and 
 iteration times.
 The returned Theta1 and Theta2 can then be used for prediction, by using

 predictPoisson.m, in the form of 
 [p, p1]= predictPoisson(Theta1, Theta2, X)
 where you put the two thetas as the model, and your predictor matrix 
 of training|validation|testing sets as X.
 The returned p is in decimal and p1 is rounded numbers. 
 p1 is guaranteed to be non-negative integers.

 calcRMSLE.m is used to evaluate how good the model is, 
 once you have done prediction on validation|testing sets.
 in the form of [cost] = calcRMSLE(pred, y)
 where you put your predicted vector as pred and true-value vector as y.
 the resultant cost is calculated as RMSLE.

 sgmdGrad.m
  and
 randInitGrad.m
  are assistant functions 
 to be quoted in the cost function, prediction and the main function.

 ParaSlectPoisson.m
 in the form of
 [opti_hidden_layer_size, opti_lambda, RMSLEvalMatrix, ...
        RMSLEtrainMatrix] = ParaSlectPoisson(X, y, Xval, yval, ...
        layer_size_list, lambda_list, miniIter)

 can be used to select an optimal size of the hidden layer, and 
 to select an optimal lambda coefficient.
 The user can specify a list for each of the two parameters to input
 or they will be automatically decided by the function.
 The user should also specify an integer for miniIter,
 which decides how many iterations are in the optimization.
 !!! Note if you specify miniIter=1000, 
 the function will actually run 6500 iterations for testing, and 
 return the results based on 2500 iterations for each combination.
 IF you don't specify a value for miniIter, a value 500 will be used,
 running 3250 iterations for test and give result based on 1250 iterations.

 Take your time play with it. Wish you good luck.
