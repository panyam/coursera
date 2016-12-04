function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
diff = (X * Theta' - Y) .* R;
J = sum(sum(diff .^ 2)) / 2;

X_grad = (diff * Theta) + (lambda * X);
Theta_grad = (diff' * X) + (lambda * Theta);

% Regularization component
J += (lambda / 2) * (sum(sum(Theta .^ 2)) + sum(sum(X .^ 2)));

grad = [X_grad(:); Theta_grad(:)];

end
