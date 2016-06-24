function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
J = 0;

hypothesis=X*theta;
for i=1:m
    tmp = J + (hypothesis(i)-y(i)).^2;
    J = tmp;
end
J = tmp/(2*m);

end
