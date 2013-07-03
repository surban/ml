function [W, P] = pivot_regressor_training_step(W, P, data, targets)
global weight_step pivot_step pivot_iters;

n_samples = size(data,2);

%%% calculate weights
%W = train_kernel_regressor_with_pivots(data, P, targets);

%% optimize weights
K = kernel_matrix(P, data);
dLdW = -(targets - W*K) * K.';
W = W - (weight_step/n_samples) * dLdW;

%% optimize pivot points
for i=1:pivot_iters
    % calulate gradient wrt pivot points
    K = kernel_matrix(P, data);
    %L = norm(targets - W * K, 'fro')^2;
    dLdK = - 2 * W.' * (targets - W*K);
    dLdP = kernel_gradient_wrt_p(K, P, data, dLdK);

    if max(max(dLdP)) > 10^5
        fprintf('gradient explosion\n');
        keyboard;
    end
    
    % gradient descent on pivot points
    P = P - (pivot_step/n_samples) * dLdP;
end
    
end