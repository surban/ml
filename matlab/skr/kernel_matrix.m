function K = kernel_matrix(x, y)
% Computes the kernel matrix for the given data. 
% Inputs:
% x - Data. Each column is a training sample. Each row is a
%     feature.
% y - Data. Each column is a training sample. Each row is a
%     feature.
% Outputs:
% K - Kernel matrix. Each column is the result of kernel evaluation of one 
%     sample of y with all samples of x. Thus K_ij = K(x_i, y_j).
% Global variables:
% kernel_l - length scale of the squared exponential kernel

global kernel_l

%%
% test data:
%x=[ 11 21 31;  12 22 32;];
%y=[ 101 201; 102 202;];

n_features = size(x, 1);
assert(size(y, 1) == n_features, 'x and y must have equal number of features');
n_samples_x = size(x, 2);
n_samples_y = size(y, 2);

xe = reshape(x', 1, n_samples_x, n_features);
kx = repmat(xe, [n_samples_y, 1, 1]);

ye = reshape(y', n_samples_y, 1, n_features);
ky = repmat(ye, [1, n_samples_x, 1]);

d = sum((kx-ky).^2, 3);
K = exp(-d / (2*kernel_l^2));

K=K';

end