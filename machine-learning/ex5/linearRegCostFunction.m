function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
Hx = (theta' * X')';
thetas_without_0 = theta(2:size(theta, 1));

reg_thetas = (lambda / 2) * sum(thetas_without_0 .* thetas_without_0) / m;
J = (sum((Hx - y) .* (Hx - y)) / (2 * m)) + reg_thetas;

beta = (Hx - y);
p1 = (X' * beta) / m;
grad = p1 + ((lambda * theta) / m);
grad(1) = p1(1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
