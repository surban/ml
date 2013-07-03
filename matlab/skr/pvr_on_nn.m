global kernel_l;

%% parameters
use_gpu = gpuDeviceCount;
rate_fac = 1.1;
kernel_l = 2;

%% load data
load('boston.mat');
data = boston.data';
targets = boston.target';

% normalize data
maxdata = max(data,[],2);
maxdata(maxdata == 0) = 1;
data = data ./ repmat(maxdata, 1, size(data,2));

% add bias
data = [data; ones(1, size(data,2))];

%% initialize variables 
n_samples = size(data,2);
n_pivs = floor(n_samples / 5);
n_features = size(data,1);
n_targets = size(targets,1);
n_hidden = n_features;

%% initialize weights and pivots
W = rand(n_targets, n_pivs) - 0.5;
V = rand(n_hidden, n_features) - 0.5;
P = rand(n_hidden, n_pivs) - 0.5;

%% copy to GPU
if use_gpu
    W = gpuArray(W);
    P = gpuArray(P);
    V = gpuArray(V);
    data = gpuArray(data);
    targets = gpuArray(targets);
end

%% iterate
n_iters = 1000000;
last_obj = 100000000;
rate = 0.0001;
rate_hold = 1000;

fprintf('Start:\n');
for iter=1:n_iters
   
    %% calculate hiddens H and kernel matrix K
    H = sigmoid(V * data);   
    dsigmoid = (1-H) .* H;
    K = kernel_matrix(P, H);        
    
    %% calculate performance
    tar_err = norm(W*K - targets, 'fro');    
    fprintf('iter=%07d rate=%.5g tar_err=%.5f\n', ...
        iter, rate, tar_err);
   
    %% adjust learning rate
    obj = tar_err;   
    if obj > last_obj
        rate = rate / rate_fac;
        rate_hold = 100;
        
        % revert
        W = last_W;
        P = last_P;
        V = last_V;
        obj = last_obj;
    elseif rate_hold < 0
        rate = rate * rate_fac;        
    end
    rate = min(0.01, rate);
    rate_hold = rate_hold - 1;    

    last_W = W;
    last_V = V;
    last_P = P;    
    last_obj = obj;          
    
    %% gradient wrt W and P
    [dLdW, dLdP] = pivot_regressor_gradients(K, W, P, H, targets);      

    %% gradient wrt V
    dLdK = - 2 * W.' * (targets - W*K);
    dLdH = kernel_gradient_wrt_x(K, P, H, dLdK);    
    dLdV = (dLdH .* dsigmoid) * data.';

    %% update parameters
    W = W - (rate/n_samples) * dLdW;
    P = P - (rate/n_samples) * dLdP;
    V = V - (rate/n_samples) * dLdV;       
    
end

%% final error
fprintf('Resulting error with %d hidden units and %d pivots: %g\n', ...
    n_hidden, n_pivs, tar_err);


