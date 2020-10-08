function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

temp = theta;
temp(1) = 0;

hx = sigmoid(X * theta);
J = sum(-y' * log(hx) - (1 - y') * log(1 - hx)) / m + ...
lambda / (2*m) * sum(temp .^ 2);

grad = X' * (hx - y) / m + lambda / m .* temp;


% =============================================================

grad = grad(:);

end
