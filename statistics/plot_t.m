clf;
alpha = 0.05;
n=2:0.1:10;
plot(n,tinv(1-alpha/2, n-1));

title(sprintf('\\alpha/2 fractiles of the t_{n-1} distribution with \\alpha = %g', alpha));
xlabel('Number of samples n');
ylabel('\alpha/2 fractile of t_{n-1}');

