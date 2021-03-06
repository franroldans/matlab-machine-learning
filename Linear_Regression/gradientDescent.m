function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    tmp0 = 0;
    tmp1 = 0;
    
    %Compute the derivative term to obtain desired theta:
    for i=1:m
        tmp0 = tmp0 + (theta' * X(i,:)' - y(i)); 
        tmp1 = tmp1 + (theta' * X(i,:)' - y(i)) * X(i,2);
    end
    theta(1) = theta(1) - alpha * tmp0 / m;
    theta(2) = theta(2) - alpha * tmp1 / m;

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
end
