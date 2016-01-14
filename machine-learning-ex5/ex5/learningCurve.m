function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation error, you should instead evaluate on
%       the _entire_ cross validation set (Xval and yval).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i) and error_val(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------
       % Now, costFunction is a function that takes in only one argument
       options = optimset('MaxIter', 200, 'GradObj', 'on');


       for i = 1:m
           % Compute train/cross validation errors using training examples 
           xx =  X(1:i, :); 
           yy = y(1:i);  
          
           
           % Initialize Theta for train subset
           initial_theta_t = zeros(size(xx, 2), 1); 
           
           % Create "short hand" for the cost function to calculate train theta
           costFunction_t = @(t) linearRegCostFunction(xx, yy, t, 0);

           % Minimize using fmincg
           theta_t = fmincg(costFunction_t, initial_theta_t, options);
           
           error_train(i) = (1/(2*m)) * sum((xx*theta_t - yy ).^2);
           
           initial_theta_cv = zeros(size(Xval, 2), 1);
           
           % cost function to calculate cv theta
           costFunction_cv = @(s) linearRegCostFunction(Xval, yval, s, 0);
           
           theta_cv = fmincg(costFunction_cv, initial_theta_cv, options);
           
           error_val(i) = (1/(2*m)) * sum((Xval*theta_t - yval ).^2);
           
           
       end





% -------------------------------------------------------------

% =========================================================================

end
