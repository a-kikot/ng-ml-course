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

% one-hot encoding;
Y = y == 1:max(y);

% FP
a1 = [ones(size(X), 1), X];
z2 = a1 * Theta1'; 
a2 = sigmoid(z2);
a2 = [ones(size(a2), 1), a2];
z3 = a2 * Theta2'; 
a3 = sigmoid(z3); 
hx = a3; % [5000 10]

% computing cost function
J = sum(sum(-Y .* log(hx) - (1 - Y) .* log(1 - hx))) / m;

reg_ = lambda * (sum(Theta1(1:end, 2:end)(:).^2) ...
	+ sum(Theta2(1:end, 2:end)(:).^2)) / (2*m);
J = J + reg_;

% BP 
d3 = a3 - Y; 
d2 = (d3 * Theta2 .* sigmoidGradient([ones(size(z2),1),z2]))(:, 2:end);

D2 = (d3' * a2);
D1 = (d2' * a1);

Theta1_grad = D1 ./ m + lambda / m * [zeros(size(Theta1), 1) Theta1(:, 2:end)];
Theta2_grad = D2 ./ m + lambda / m * [zeros(size(Theta2), 1) Theta2(:, 2:end)];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
