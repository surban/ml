function FG = kernel_gradient_wrt_x(K, p, x, FKG)
% Calculates the gradient of a kernel matrix K(p,y) contained in a scalar 
% function w.r.t. to the input x.
% Inputs:
% K  - Kernel matrix.
% p  - Input (pivot points).
% x  - Input.
% FKG- Gradient of scalar function w.r.t. the kernel matrix.
% Outputs:
% FG - Gradient of scalar function w.r.t. input x.

global kernel_l;
FG = 1/kernel_l^2 .* (p*(K.*FKG) - x * diag(diag(FKG.' * K)));

end
