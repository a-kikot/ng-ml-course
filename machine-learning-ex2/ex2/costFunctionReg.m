function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


Hx = sigmoid(X * theta);

J = (- y' * log(Hx) - (1 - y') * log(1 - Hx)) / m + ...
lambda / (2 * m) * sum(theta(2:size(theta)) .^ 2);


grad = X' * (Hx - y) / m;  
temp = theta;
temp(1) = 0;
grad = grad + lambda / m .* temp;


% =============================================================

end
