%% init
animate = false;

if ~animate
    rng('default');
    rng('shuffle');
end
    
clear img;

misses = 0;
naive_misses = 0;
alpha = 0.002;

for iterations = 1:1000
    %for samples = 1:100
    for samples = 5

        %% generate data
        if animate
            rng('default');
            %rng(77); % 2 is bad
            rng('shuffle');
        end
        mu = 0;
        sigma = 1;
        r = mu + sigma * randn(1, samples);

        %% estimate
        [muhat, sigmahat, muci, sigmaci] = normfit(r, alpha);
        fprintf('mu       = %f +/- %f\n', muhat, muhat - muci(1));
        %fprintf('sigma = %f\n', sigmahat);

        %% my estimate
        M = mean(r);
        V = std(r).^2;
        t = tinv(1-alpha/2, samples - 1);
        %my_pm = t * sqrt(V / samples);
        my_pm = t * std(r) / sqrt(samples);
        fprintf('my mu    = %f +/- %f\n', M, my_pm);

        %% standard deviation of sample mean
        mean_conf = 3 * std(r) / sqrt(samples);
        fprintf('naive mu = %f +/- %f\n', M, mean_conf);

        %% check
        if ~(muci(1) <= 0 && 0 <= muci(2))
            fprintf('miss\n')
            misses = misses + 1;
        end
        
        if ~(M - mean_conf <= 0 && 0 <= M + mean_conf)
            fprintf('naive miss\n');
            naive_misses = naive_misses + 1;
        end

        %% plot
        %figure(1);
        %hist(r);

        figure(2); 
        clf;
        hold on;
        plot(r, zeros(size(r)), 'rx', 'MarkerSize', 10);
        axis([-5 5 -0.1 0.1]);

        line([muci(1) muci(2)], [-0.02 -0.02], 'Color', 'blue');
        line([muhat muhat], [-0.05 0.05], 'Color', 'blue');
        line([0 0], [-0.05 0.05], 'Color', 'black', 'LineStyle', ':');
        line([muhat-mean_conf muhat+mean_conf], [0.02 0.02], 'Color', 'red');

        drawnow;
        img(samples) = getframe;
    end
    
end

%% show results
if ~animate
    fprintf('Misses: %d / %d (should be not more than %f)\n', ...
        misses, iterations, alpha * iterations);
    fprintf('Naive misses: %d / %d\n', naive_misses, iterations);
else
    %movie(img,1,3);
end
