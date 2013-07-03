function ty = predict_kernel_regressor(w, x, tx)
% Predicts targets using a kernel regressor.
% Inputs:
% w - Weights.
% x - Training data. Each column is a training sample. Each row is a
%     feature.
% tx- Test data. Each column is a sample. Each row is a
%     feature.
% Outputs:
% ty- Predictions.

K = kernel_matrix(tx, x);
ty = w * K;

end
