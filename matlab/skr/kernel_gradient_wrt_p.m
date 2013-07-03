function FG = kernel_gradient_wrt_p(K, p, x, FKG)
% Calculates the gradient of a kernel matrix K(p,y) contained in a scalar 
% function w.r.t. to the pivots p.
% Inputs:
% K  - Kernel matrix.
% p  - Input (pivot points).
% x  - Input.
% FKG- Gradient of scalar function w.r.t. the kernel matrix.
% Outputs:
% FG - Gradient of scalar function w.r.t. input p.

global kernel_l;
FG = 1/kernel_l^2 .* (x * ((K.*FKG).') - p * diag(diag(FKG * K.')));



% numerical computation for testing:
% FG = [];
% for n=1:size(x,1)
%     for m=1:size(p,2)
%         s = 0;
%         for j=1:size(K,2)
%             s = s + FKG(m,j) * K(m,j) * (p(n,m) - x(n,j));
%         end
%         FG(n,m) = - s / kernel_l^2;
%     end
% end
%
% FG = 1/kernel_l^2 .* (x * ((K.*FKG).'));
% for n=1:size(x,1)
%     for m=1:size(p,2)
%         s = 0;
%         for j=1:size(K,2)
%             s = s + FKG(m,j) * K(m,j) * p(n,m);
%         end
%         FG(n,m) = FG(n,m) - s / kernel_l^2;
%     end
% end


end
