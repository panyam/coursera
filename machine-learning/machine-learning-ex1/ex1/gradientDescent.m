function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    diffs = (X * theta) - y;
    abym = alpha / m;
    oldtheta0 = theta(1);
    oldtheta1 = theta(2);

    newtheta0 = oldtheta0 - (abym * sum(diffs .* X(:,1)));
    newtheta1 = oldtheta1 - (abym * sum(diffs .* X(:,2)));
    theta = [ newtheta0 ; newtheta1 ];

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end

end
