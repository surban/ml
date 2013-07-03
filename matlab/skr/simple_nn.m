
%% parameters
rate_fac = 1.1;
use_gpu = true;

%% load data
load('boston.mat');
data = boston.data';
targets = boston.target';

% data=data(:,1:10);
% targets=targets(:,1:10);

% normalize data
maxdata = max(data,[],2);
maxdata(maxdata == 0) = 1;
data = data ./ repmat(maxdata, 1, size(data,2));

% add bias
data = [data; ones(1, size(data,2))];

%% initialize variables 
n_samples = size(data,2);
n_features = size(data,1);
n_targets = size(targets,1);
n_hidden = n_features;
n_hidden = 200;

%% initialize weights 
W = rand(n_targets, n_hidden) - 0.5;
V = rand(n_hidden, n_features) - 0.5;

%% copy to GPU
if use_gpu
    W = gpuArray(W);
    V = gpuArray(V);
    data = gpuArray(data);
    targets = gpuArray(targets);
end

%% iterate
n_iters = 1000000;
%n_iters=1;
last_obj = 100000000;
rate = 0.0001;
rate_hold = 1000;

fprintf('Start:\n');
for iter=1:n_iters
   
    %% calculate hiddens H
    H = sigmoid(V * data);   
    dsigmoidH = (1-H) .* H;
    
    %% calculate outputs T
    T = W * H;
    
    %% calculate performance
    tar_err = norm(T - targets, 'fro');   
    if mod(iter, 100) == 0
        fprintf('iter=%07d rate=%.5g tar_err=%.5f\n', ...
            iter, rate, tar_err);
    end

    %% adjust learning rate
    obj = tar_err;   
    if obj > last_obj
        rate = rate / rate_fac;
        rate_hold = 100;
        
        % revert
        W = last_W;
        V = last_V;
        obj = last_obj;
    elseif rate_hold < 0
        rate = rate * rate_fac;        
    end
    rate = min(0.001, rate);
    %rate = max(0.00001, rate);
    rate_hold = rate_hold - 1;    
    
    last_obj = obj;             
    last_W = W;
    last_V = V;    
    
    %% gradients   
    % wrt. top layer
    dLdT = - 2 * (targets - T);
    
    % wrt. hidden layer
    dLdH = W.' * dLdT;
    
    % wrt. top weights W
    dLdW = dLdT * H.';
    
    % wrt. hidden weights V
    dLdV = (dLdH .* dsigmoidH) * data.';     

    %% update parameters
    W = W - (rate/n_samples) * dLdW;
    V = V - (rate/n_samples) * dLdV;       
   
end

%% final error
fprintf('Resulting error with %d hidden units: %g\n', ...
    n_hidden, tar_err);


