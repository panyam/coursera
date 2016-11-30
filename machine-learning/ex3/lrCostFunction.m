function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% STheta = size(theta)
% SX = size(X)
% SY = size(y)

% You need to return the following variables correctly 
TTX = (theta' * X')';
p1 = -y .* log(sigmoid(TTX));
p2 = (1-y) .* log(1 - sigmoid(TTX));
sums = sum(p1 - p2) / m;

thetas_without_0 = theta(2:size(theta, 1));
reg_thetas = (lambda / 2) * sum(thetas_without_0 .* thetas_without_0) / m;
J = sums + reg_thetas;

beta = sigmoid(TTX) - y;
p1 = (X' * beta) / m;
grad = p1 + ((lambda * theta) / m);
grad(1) = p1(1);

grad = grad(:);

end
