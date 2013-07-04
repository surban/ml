% R - Training
% T - Testing
% V - Validation
% X - Input
% Z - Target

% kernel: Gauss, ARD (automatic relevance determination)
% results: MSE on test set, average negative log likelihood on test set for each point separately and all together

load('../datasets/abalone_split.mat');

do_optim = true;
optim_steps = 15;

trn_x = [RX.'; VX.'];
trn_z = [RZ.'; VZ.'];
tst_x = TX.';
tst_z = TZ.';

%% GP setup
meanfunc = @meanConst; 
hyp.mean = 0;

covfunc = @covSEiso; 
ell = 1;
sf = 1;
hyp.cov = [log(ell); log(sf)];

likfunc = @likGauss; 
sn = 0.1; 
hyp.lik = log(sn);

%% minimize marginal likelihood w.r.t. hyperparameters
if do_optim
    tic;
    hyp = minimize(hyp, @gp, -optim_steps, ...
        @infExact, meanfunc, covfunc, likfunc, ...
        trn_x, trn_z);
    toc;

    %save('abalone_gp.mat', 'meanfunc', 'covfunc', 'likfunc', 'hyp');    
end


%% calculate marginal likelihood
tr_lml = gp(hyp, @infExact, ...
    meanfunc, covfunc, likfunc, ...
    trn_x, trn_z);  

%% predict
[tst_pz, tst_pv] = ...
    gp(hyp, @infExact, ...
    meanfunc, covfunc, likfunc, ...
    trn_x, trn_z, tst_x);         

mse = mean((tst_z - tst_pz).^2);
fprintf('mean squared error on test set = %f\n', mse);

%% calculate log predictive probabilities
[tst_pz, tst_pv, tst_lp] = ...
    gp(hyp, @infExact, ...
    meanfunc, covfunc, likfunc, ...
    trn_x, trn_z, tst_x, tst_z);      

avg_tst_lp = mean(tst_lp);
fprintf('mean predictive log probability on test set = %f\n', avg_tst_lp);






            