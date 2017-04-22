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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Forward Propogation  Compute H(x)
% Add ones to the X data matrix
% 5000 * 401
a1 = [ones(m, 1) X];

% second layer, hidden layer size:5000 * 25
a2 = sigmoid(a1 * Theta1');

% 5000 * 26
a2 = [ones(m, 1) a2];

% H(x) output layer ,size:5000 * 10
a3 = sigmoid(a2 * Theta2');


% Convert y from size of 5000 * 1 to that of 5000 * 10 
% 5000 * 10
new_y = zeros(m, num_labels);

for i = 1 : m
	tmp = zeros(1, num_labels);
	tmp(1, y(i)) = 1;
	new_y(i,:) = tmp;
end
% 5000 * 10
y = new_y;

% Compute cost function J
for i = 1 : num_labels
	tmp = sum(-y(:,i) .* log(a3(:,i)) - (1 - y(:,i)) .* log(1 - a3(:,i)));
	J += tmp;
end

J = J / m;

% Compute the final regularized cost function

% Ignore the bias of Theta1 & Theta2
% Theta1 size : 25 * 401 
% Theta2 size : 10 * 26
Theta1(:,1) = 0;
Theta2(:,1) = 0;

J = J + (sum(sum(Theta1.^2)) + sum(sum(Theta2.^2))) * lambda/(2 * m);

% delta_2 size : 5000 * 26
delta_2 = zeros(size(a2));
% delta_3 size : 5000 * 10
delta_3 = zeros(size(a3));

delta_3 = a3 - y;

z2 = a1 * Theta1';
z2 = [ones(m, 1) z2];
% Size: 5000* 26
delta_2 =  (delta_3 * Theta2) .* sigmoidGradient(z2);
% Remove the first column vector, new size: 5000 * 25
delta_2(:,1) = [];


Theta1_grad = (delta_2' * a1)/m + Theta1 * (lambda / m);
Theta2_grad = (delta_3' * a2)/m + Theta2 * (lambda / m);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
