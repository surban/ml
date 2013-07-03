%% parameters
global kernel_l

kernel_l = 1;
hidden_sizes = [ 3 ];


%% load data
load('boston.mat');
data = boston.data';
targets = boston.target';

data=data(:,1:10);
maxdata = max(data,[],2);
maxdata(maxdata == 0) = 1;
data = data ./ repmat(maxdata, 1, size(data,2));
targets=targets(:,1:10);

%% initialize variables 
n_samples = size(data,2);
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

%% initialize weights
W = {};

for l=1:top_layer
    W{l} = [];
end

%% iterate
n_iters = 1;
for i=1:n_iters   

    %% W-step: learn kernel regressors layerwise
    for l=2:top_layer
        W{l} = train_kernel_regressor(z{l-1}, z{l});      
    end

    %% test
    K{1} = kernel_matrix(z{1}, z{1});
    K{2} = kernel_matrix(z{2}, z{2});    
    %break
    
    %% calculate error
    K{2} = kernel_matrix(z{2}, z{2});
    z3p = W{3} * K{2};
    z3d = z3p - z{3};
    err = norm(z3d, 'fro');
    fprintf('Regression error of targets: %f\n', err);
    break
    
    %% Z-step: update targets
    assert(n_hidden_layers == 1, 'only one hidden layer supported for now');

    z_iters = 5;
    z_eta = 0.001;
   
    mu = 0.1;
    mu_fac = 2;
    
    K = {};
    K{1} = kernel_matrix(z{1}, z{1});
    fprintf('Starting optimization\n');
    for Zi=1:z_iters
        % calculate error
        K{2} = kernel_matrix(z{2}, z{2});
        z3p = W{3} * K{2};
        z3d = z3p - z{3};
        err = norm(z3d, 'fro');
        fprintf('Regression error of targets: %f\n', err);
        
        % update
        K{2} = kernel_matrix(z{2}, z{2});
        dLdK2 = - W{3}.' * (z{3} - W{3} * K{2});
        dLdz2 = kernel_gradient(K{2}, z{2}, dLdK2) + mu * (z{2} - W{2} * K{1});
        
        fprintf('Magnitude of gradient: %f\n', norm(dLdz2, 'fro'));
        
        z{2} = z{2} - z_eta * dLdz2;
    end

end



