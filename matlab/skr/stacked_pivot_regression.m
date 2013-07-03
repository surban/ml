%% parameters
global kernel_l pivot_step weight_step pivot_iters;

pivot_iters = 1;

mu = 0.1;
mu_fac = 1.1;
rate_fac = 1.1;

kernel_l = 2;
hidden_sizes = [ 50 ];


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

%% initialize variables 
n_samples = size(data,2);
n_pivs = floor(n_samples / 5);
n_pivs = 10;
n_features = size(data,1);
n_hidden_layers = numel(hidden_sizes);
top_layer = n_hidden_layers+2;

%% initialize auxilary coordinates
z = {};

% bottom layer is given by inputs
z{1} = data;

% intermediate targets are initialized randomly
for i=1:n_hidden_layers
    z{i+1} = rand(hidden_sizes(i), n_samples) - 0.5;
end

% top layer is given by targets
z{top_layer} = targets;

%% initialize weights and pivots
W = {};
P = {};

for l=1:top_layer
    if l > 1
        W{l} = rand(size(z{l}, 1), size(P{l-1}, 2)) - 0.5;   
    end
    pivs = randsample(n_samples, n_pivs);
    P{l} = z{l}(:,pivs);
end

%% iterate
n_iters = 1000;
last_obj = 100000000;
rate = 0.001;
rate_hold = 1;
fprintf('Start:\n');
while mu < 100

    last_W = W;
    last_P = P;
    last_z = z;
    
    z_eta = rate;   
    pivot_step = rate;
    weight_step = rate;
    
    
    %% W-step: learn kernel regressors layerwise
    for l=2:top_layer     
        %fprintf('layer=%d\n', l);
        [W{l}, P{l-1}] = pivot_regressor_training_step(W{l}, P{l-1}, z{l-1}, z{l});      
    end

    %continue;
    
    %% Z-step: update targets
    %fprintf('Z-step:\n');   
    assert(n_hidden_layers == 1, 'only one hidden layer supported for now');
   
    K = {};
    K{1} = kernel_matrix(P{1}, z{1});
    
    % calculate prediction error
    K{2} = kernel_matrix(P{2}, z{2});
    z3p = W{3} * K{2};
    z3d = z3p - z{3};
    tar_err = norm(z3d, 'fro');

    % calculate layer discrepancy
    z2p = W{2} * K{1};
    z2d = z2p - z{2};
    discp = norm(z2d, 'fro');

    % calculate objective function
    obj = 1/2 * norm(z{3} - W{3}*K{2}, 'fro')^2 + ...
        mu/2 * norm(z{2} - W{2}*K{1}, 'fro')^2;

    fprintf('rate=%.5g mu=%.5f  obj=%.5f  tar_err=%.5f  discp=%.5f\n', ...
        rate, mu, obj, tar_err, discp);

    % gradient w.r.t. intermediates z{2}
    %K{2} = kernel_matrix(P{2}, z{2});
    dLdK2 = - W{3}.' * (z{3} - W{3} * K{2});
    dLdz2 = kernel_gradient_wrt_x(K{2}, P{2}, z{2}, dLdK2) + ...
        mu * (z{2} - W{2} * K{1});

    %fprintf('Magnitude of gradient: %f\n', norm(dLdz2, 'fro'));

    % update
    z{2} = z{2} - (z_eta/n_samples) * dLdz2;
    
    % check for convergence
    if abs(obj - last_obj) < 0.01
        fprintf('converged\n');
        mu = mu * mu_fac;
        last_obj = 100000000;
        obj = last_obj - 1;
        rate = rate / rate_fac;
        %break;
    end
    
    if obj > last_obj
        rate = rate / rate_fac;
        rate_hold = 100;
        
        W = last_W;
        P = last_P;
        z = last_z;
        obj = last_obj;
    elseif rate_hold < 0
        rate = rate * rate_fac;        
    end
    rate_hold = rate_hold - 1;
    
    last_obj = obj;          
end

%% final W-step: learn kernel regressors layerwise
%for l=2:top_layer        
%    [W{l}, P{l-1}] = pivot_regressor_training_step(P{l-1}, z{l-1}, z{l});      
%end

%% calculate final values of layers
for l=2:top_layer
    K{l-1} = kernel_matrix(P{l-1}, z{l-1});
    z{l} = W{l} * K{l-1};
end

%% calculate prediction error
dt = z{top_layer} - targets;
err = norm(dt, 2);
fprintf('Resulting error with %d pivots: %g\n', n_pivs, err);


