function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;
return;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);

%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

c_vals = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000];

simma_vals = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000];


optimal_error =10^10;

 for c_val = c_vals
 for sigma_val = simma_vals
      model= svmTrain(X, y, c_val, @(x1, x2) gaussianKernel(x1, x2, sigma_val)); 
      predictions = svmPredict(model, Xval);
      pred_error = mean(double(predictions ~= yval));
     % visualizeBoundary(X, y, model);
     % puse;
      if(pred_error < optimal_error)
        optimal_error = pred_error;
        C = c_val;
        sigma = sigma_val;
      end
 end
 end



% =========================================================================

c_val = [];
end
