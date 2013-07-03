global kernel_l;

kernel_l = 10;

%% test gradient
x = [11 21 31 41 51; 
     12 22 32 42 55;];
K = kernel_matrix(x,x);

f = sum(sum(K));
FKG = ones(size(K));

FG = kernel_gradient(K, x, FKG);

% calculate gradient numerically
FGnum = [];
epsilon = 0.001;
for i=1:size(FG,1)
    for j=1:size(FG,2)
        xt = x;
        xt(i,j) = xt(i,j) + epsilon;
        Kt = kernel_matrix(xt, xt);
        ft = sum(sum(Kt));       
        FGnum(i,j) = (ft - f) / epsilon;              
    end
end

FG
FGnum

assert(all(all(FG-FGnum < 0.01)));
