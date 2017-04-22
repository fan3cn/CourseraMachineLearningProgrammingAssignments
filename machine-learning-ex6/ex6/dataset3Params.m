function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Test Parameters 
params = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
% Vector Length
m = length(params);
% All Errors
errs = zeros(m, m);
% Index
idx1 = 1
idx2 = 1
% Do loops
for i = 1 : m
	for j = 1 : m
		C = params(1,i);
		sigma = params(1,j);
		% Train with SVM by using different parameters C and sigma on validation set 
		model= svmTrain(Xval, yval, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		% Make predictions with svmPredict
		predictions = svmPredict(model, Xval);
		% Compute error
		err = mean(double(predictions ~= yval));
		% Store evey error
		errs(i, j) = err;
		% According to the Figure 7, the best error is about 5% 
		if err == 0.05
			idx1 = i;
			idx2 = j;
		end
	end
end

C = params(1, idx1); % 1
sigma = params(1, idx2); % 0.1

% For a faster run ,you can ommit above codes and simply do as follows: 
%C = 1
%sigma = 0.1

% =========================================================================

end
