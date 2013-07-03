function w = train_kernel_regressor(x, y)
% Trains a kernel regressor.
% x - Input data. Each column is a training sample. Each row is a
%     feature.
% y - Targets. Each column is a training sample. Each row are the target
%     values for an output neuron.

K = kernel_matrix(x,x);
%w = y * pinv(K);
w = y / K;

end