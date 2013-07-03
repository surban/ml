function FG = kernel_gradient(K, x, FKG)
% Calculates the gradient of a kernel matrix K(x,x) contained in a scalar 
% function w.r.t. the input x.
% Inputs:
% K  - Kernel matrix.
% x  - Input.
% FKG- Gradient of scalar function w.r.t. the kernel matrix.
% Outputs:
% FG - Gradient of scalar function w.r.t. input

global kernel_l;
FG = 1/kernel_l^2 .* (x*(K.*(FKG+FKG.')) - x*diag(diag(FKG*K + FKG.'*K)));


% numerical computation for testing:
% FG = [];
% %FG = 1/kernel_l^2 * (x*(K.*(FKG+FKG.')));
% 
% for i=1:size(x,1)
%     for j=1:size(x,2)
%         s = 0;
%         for n=1:size(x,2)
%             s = s + (FKG(n,j) + FKG(j,n)) * K(n,j) * x(i,n);            
%         end
%         FG(i,j) = s / kernel_l^2;
%     end
% end
% 
% 
% FG = 1/kernel_l^2 * (x*(K.*(FKG+FKG.')));
% 
% nd = [];
% for i=1:size(x,1)
%     for j=1:size(x,2)
%         s = 0;
%         nd(j) = 0;
%         for n=1:size(x,2)
%             s = s - (FKG(n,j) + FKG(j,n)) * K(n,j) * x(i,j);            
%             nd(j) = nd(j) + (FKG(n,j) + FKG(j,n)) * K(n,j);
%         end
%         FG(i,j) = FG(i,j) + s / kernel_l^2;
%     end
% end
% 
% nd
% sd = FKG*K + FKG.'*K


% for i=1:size(x,1)
%     for j=1:size(x,2)
%         s = 0;
%         for n=1:size(x,2)
%             s = s + (FKG(n,j) + FKG(j,n)) * K(n,j) * (x(i,n) - x(i,j));            
%         end
%         FG(i,j) = s / kernel_l^2;
%     end
% end


end
