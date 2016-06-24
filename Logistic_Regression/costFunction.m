function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta)); % it should have the same dimension as theta
tmp0 = 0;
hypothesis = sigmoid(X*theta); % Use sigmoid function as hypothesis function.

for i=1:m
   tmp0 = tmp0 + y(i)*log(hypothesis(i))+(1-y(i))*log(1-hypothesis(i));
end
J=-(tmp0/m);
clear tmp
for i=1:size(theta)
    grad(i) = (hypothesis-y)'* X(:,i) ./ m;
end

end
