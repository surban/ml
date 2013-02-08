a_mean = 1;
a_sigma = 1;
b_mean = 1.3;
b_sigma = a_sigma;

n_samples = 100;

a = a_mean + a_sigma * randn(n_samples, 1);
b = b_mean + b_sigma * randn(n_samples, 1);

alpha = 0.05;

%% plot
% figure(1);
% hist(a);
% 
% figure(2);
% hist(b);

figure(2);
boxplot([a b]);

%% difference
d = b-a;
figure(1);
hist(d);
[muhat, sigmahat, muci, sigmaci] = normfit(d, alpha);
fprintf('mu       = %f +/- %f\n', muhat, muhat - muci(1));

%% test
[h,p] = ttest(d, 0, alpha);
fprintf('h: %d  p: %f\n', h, p);



