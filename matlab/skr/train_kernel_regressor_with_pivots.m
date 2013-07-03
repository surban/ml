function w = train_kernel_regressor_with_pivots(x, p, y)
% Trains a kernel regressor.
% x - Input data. Each column is a training sample. Each row is a
%     feature.
% p - Pivot points. Each column is a training sample. Each row is a 
%     feature.
% y - Targets. Each column is a training sample. Each row are the target
%     values for an output neuron.

lastwarn('');

K = kernel_matrix(p, x);
%w = y * pinv(K);
w = y / K;

[~, warn_id] = lastwarn;
if strcmp(warn_id, 'MATLAB:rankDeficientMatrix')
    keyboard;
end

end