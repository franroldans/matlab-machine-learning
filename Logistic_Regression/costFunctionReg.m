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
tmp0 = 0;
tmp1 = 0;
tmp3 = 0;

%Compute cost function sum:
for i=1:m
   tmp0 = tmp0 + y(i)*log(hypothesis(i))+(1-y(i))*log(1-hypothesis(i));
end
J_aux=-(tmp0/m);

%Compute regularization term sum:
for j=2:size(theta)
    tmp1 = tmp1 + theta(j)^2;
end
J = J_aux + (lambda/(2*m))*tmp1;

%Compute Gradient Descent derivative terms:
for i=1:size(theta)
   tmp3 =(hypothesis-y)'*X(:,i);
   if i==1
       grad(i) = (1/m)*tmp3;
   else
       grad(i) = (1/m)*tmp3 + (lambda/m)*theta(i);
   end
end

end
