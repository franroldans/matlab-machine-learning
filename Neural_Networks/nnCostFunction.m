function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


a1 = [ones(m, 1) X];
a2 = [ones(m, 1) sigmoid(a1 * Theta1')];
hypothesis = sigmoid(a2 * Theta2');
yVector = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);
J_aux = -yVector .* log(hypothesis) - (1 - yVector) .* log(1 - hypothesis);

Theta1NoBias = Theta1(:, 2:end);
Theta2NoBias = Theta2(:, 2:end);
J = (1 / m) * sum(sum(J_aux)) + (lambda / (2 * m)) * (sum(sum(Theta1NoBias .^ 2)) + sum(sum(Theta2NoBias .^ 2)));


delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));

for i = 1:m,
	a1Trans = a1(i,:)';
	a2Trans = a2(i,:)';
	hypothesisTrans = hypothesis(i,:)';
	yVectorTrans = yVector(i,:)';

	d3t = hypothesisTrans - yVectorTrans;

	z2t = [1; Theta1 * a1Trans];
	d2t = Theta2' * d3t .* sigmoidGradient(z2t);

	delta1 = delta1 + d2t(2:end) * a1Trans';
	delta2 = delta2 + d3t * a2Trans';
end;

Theta1ZeroBias = [ zeros(size(Theta1, 1), 1) Theta1NoBias ];
Theta2ZeroBias = [ zeros(size(Theta2, 1), 1) Theta2NoBias ];
Theta1_grad = (1 / m) * delta1 + (lambda / m) * Theta1ZeroBias;
Theta2_grad = (1 / m) * delta2 + (lambda / m) * Theta2ZeroBias;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
