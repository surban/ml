rng(100);

%% load data
data = rand(10,10) - 0.5;
targets = rand(1,10) - 0.5;

%% initialize variables 
n_samples = size(data,2);
n_features = size(data,1);
n_targets = size(targets,1);
n_hidden = 5;

%% initialize weights 
W = rand(n_targets, n_hidden) - 0.5;
V = rand(n_hidden, n_features) - 0.5;

%% calculate hiddens H
H = sigmoid(V * data);   
dsigmoidH = (1-H) .* H;

%% calculate outputs T
T = sigmoid(W * H);
dsigmoidT = (1-T) .* T;

%% calculate loss
L = norm(T - targets, 'fro')^2;

%% gradients   
% wrt. top layer
dLdT = - 2 * (targets - T);

% wrt. hidden layer
dLdH = W.' * (dLdT .* dsigmoidT);

% wrt. top weights W
dLdW = (dLdT .* dsigmoidT) * H.';

% wrt. hidden weights V
dLdV = (dLdH .* dsigmoidH) * data.';     

%% numeric gradients
epsilon = 1e-4;

NdLdW = [];
for n=1:size(W,1)
    for m=1:size(W,2)
        dW = W;
        dW(n,m) = dW(n,m) + epsilon;
        
        H = sigmoid(V * data);   
        T = sigmoid(dW * H);
        dL = norm(T - targets, 'fro')^2;
        
        NdLdW(n,m) = (dL - L) / epsilon;
    end
end
dforw = norm(NdLdW - dLdW, 'fro')

NdLdV = [];
for n=1:size(V,1)
    for m=1:size(V,2)
        dV = V;
        dV(n,m) = dV(n,m) + epsilon;
        
        H = sigmoid(dV * data);   
        T = sigmoid(W * H);
        dL = norm(T - targets, 'fro')^2;
        
        NdLdV(n,m) = (dL - L) / epsilon;
    end
end
dforv = norm(NdLdV - dLdV, 'fro')


        