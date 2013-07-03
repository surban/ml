clear;
global kernel_l weight_step pivot_step;

kernel_l = 10;
rate = 10;

load( 'boston.mat');
data = boston.data';
targets = boston.target';
n_samples = size(data,2);

%% sample initial pivot points
n_pivs = floor(n_samples/5);
%n_pivs = n_samples;
pivs = randsample(n_samples, n_pivs);

piv_data = data(:,pivs);
piv_targets = targets(:,pivs);

start_piv_data = piv_data;

%% initialize weights
W = rand(size(targets,1), size(piv_data,2)) - 0.5;

%% iterate optimization of weights and pivot points
last_L = 0;
L = 100000000;
n = 0;
prevent_step_increase = 0;
for n=1:1000
    old_W = W;
    old_piv_data = piv_data;   
    
    last_L = L;
    n = n+1;
    
    weight_step = rate;
    pivot_step = rate;
          
    % gradient descent on weights and pivots
    [W, piv_data] = pivot_regressor_training_step(W, piv_data, data, targets);
    
    % output loss
    K = kernel_matrix(piv_data, data);
    L = sqrt(norm(targets - W * K, 'fro')^2);
    fprintf('%d: (rate=%g) Prediction loss with %d pivots: %g\n', ...
        n, rate, n_pivs, L);       
    
    % adjust step
    if L > last_L
        rate = rate / 2;
        W = old_W;
        piv_data = old_piv_data;
        L = last_L;
        prevent_step_increase = 100;        
    elseif abs(last_L - L) < 0.1 && prevent_step_increase < 0 
        rate = rate * 2;
        prevent_step_increase = 100;   
    end
    prevent_step_increase = prevent_step_increase - 1;
    
end

%% calculate weights
%W = train_kernel_regressor_with_pivots(data, piv_data, targets);

%% calculate predictions with currents weights
K = kernel_matrix(piv_data, data);
pred_targets = W * K;

dt = pred_targets - targets;
err = norm(dt, 2);
fprintf('Resulting error with %d pivots: %g\n', n_pivs, err);



