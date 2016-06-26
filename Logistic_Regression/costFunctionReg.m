function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

hypothesis = sigmoid(X*theta);
temp = theta; % Alias of theta
temp(1)=0; % We don't apply regularization to theta(1).
J_aux = y.*log(hypothesis)+(1-y).*log(1-hypothesis); % Cost computation
J = -sum(J_aux)/m + (lambda/(2*m)) * sum(temp.^2);
grad = (1/m).* X' * (hypothesis-y) + (lambda/m) * temp;

grad = grad(:);

end
