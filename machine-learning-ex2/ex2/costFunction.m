function [J, grad] = costFunction(theta, X, y)


m = length(y); % number of training examples


J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta [ 3 1 ]
%


Hx = sigmoid(X * theta);
J = (- y' * log(Hx) - (1 - y') * log(1 - Hx)) / m;


grad = X' * (Hx - y) / m;



% =============================================================

end
