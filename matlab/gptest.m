meanfunc = {@meanZero}; hyp.mean = [];
covfunc = {@covSEiso}; ell = 0.6; sf = 10; hyp.cov = log([ell; sf]);
%covfunc = {@covSEisoU}; ell = 0.4; hyp.cov = log([ell]);
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);


clf
plot(x, y, '+')


nlml = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y);

z = (-6:0.1:6)';
[m s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, z);

f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)]; 
fill([z; flipdim(z,1)], f, [7 7 7]/8)
hold on; plot(z, m); plot(x, y, '+');
ylim([-5 5])
