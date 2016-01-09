function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

h_x =  sigmoid( X * theta);
first = -y .*  log( h_x);
second = (1 - y) .* log(1 - h_x);

subt = theta(2:length(theta))
J =  1/m * (sum(first - second)) + (lambda / (2*m)) * sum(subt.*subt);




% grad_0  = 1/m * X(1)' * (h_x - y)

%for j = 2: length(X)
grad = 1/m * X' * (h_x - y) + (lambda/m) * theta;
grad(1) = grad(1) - (lambda/m) * theta(1);

%end

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% =============================================================

end
