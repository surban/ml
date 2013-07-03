
global kernel_l;

kernel_l = 5;

% 2 pivots a 2 features
P = [ 11.5 21.5;
      12.5 22.5; ];
  
% 3 samples a 2 features
X = [ 11 21 31;
      12 22 32; ];
  
% 3 corresponding targets
Y = [ 101 201 301 ];
  
% kernel matrix  
K = kernel_matrix(P, X);   

rng(101);
W = rand(size(Y,1), size(K, 1));

L = norm(Y - W*K, 'fro')^2;


% %% testing kernel gradient
% 
% % analytic gradient
% dKdP = [];
% for i=1:size(K,1)
%     for j=1:size(K,2)
%         for n=1:size(P,1)
%             for m=1:size(P,2)
%                 if i == m                    
%                     dKdP(i,j,n,m) = -1/kernel_l^2 * K(i,j) * (P(n,i) - X(n,j)); 
%                 else
%                     dKdP(i,j,n,m) = 0;
%                 end
%             end
%         end
%     end
% end
% %dKdP
% 
% % numeric gradient
% NdKdP = [];
% epsilon = 0.000001;
% for i=1:size(K,1)
%     for j=1:size(K,2)
%         for n=1:size(P,1)
%             for m=1:size(P,2)
%                 DP = P;
%                 DP(n,m) = DP(n,m) + epsilon;
%                 DK = kernel_matrix(X, DP);
%                 NdKdP(i,j,n,m) = (DK(i,j) - K(i,j)) / epsilon;
%             end
%         end
%     end
% end
% %NdKdP
% 
% df = NdKdP - dKdP;
% fprintf('|df|_2 = %g\n', sqrt(sum(sum(sum(sum(df.^2))))));

%% testing loss function gradient wrt p

% analytic gradient
dLdK = - 2 * W.' * (Y - W*K);
dLdP = kernel_gradient_wrt_p(K, P, X, dLdK);

% numeric gardient
epsilon = 0.00001;
NdLdP = [];
for i=1:size(P,1)
    for j=1:size(P,2)
        DP = P;
        DP(i,j) = DP(i,j) + epsilon;
        DK = kernel_matrix(DP, X);
        DL = norm(Y - W*DK, 'fro')^2;
        NdLdP(i,j) = (DL - L) / epsilon;
    end
end
        

dLdP
NdLdP


dfl = NdLdP - dLdP;
fprintf('|dfl|_2 = %g\n', norm(dfl, 'fro'));


%% testing loss function gradient wrt x

% analytic gradient
dLdK = - 2 * W.' * (Y - W*K);
dLdX = kernel_gradient_wrt_x(K, P, X, dLdK);

% numeric gardient
epsilon = 0.00001;
NdLdX = [];
for i=1:size(X,1)
    for j=1:size(X,2)
        DX = X;
        DX(i,j) = DX(i,j) + epsilon;
        DK = kernel_matrix(P, DX);
        DL = norm(Y - W*DK, 'fro')^2;
        NdLdX(i,j) = (DL - L) / epsilon;
    end
end
        

dLdX
NdLdX


dfl = NdLdX - dLdX;
fprintf('|dfl|_2 = %g\n', norm(dfl, 'fro'));

%% product test
% on = [];
% for n=1:size(P,1)
%     for m=1:size(P,2)
%         on(n,m) = 0;
%         for j=1:size(K,2)
%             on(n,m) = on(n,m) + dLdK(m,j) * K(m,j) * P(n,m);
%         end
%     end
% end
% 
% oa = P * diag(diag(dLdK * K.'));
% 
% oa
% on


