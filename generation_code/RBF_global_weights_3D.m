function fc = RBF_global_weights_3D (xd,yd,zd,fd,xc,yc,zc, varargin)
% Computes the radial basis function interpolation
% Input parameters
% xd,yd,zd  Column vectors with node locations where data is present
%    fd     The data values at each point xd,yd,zd
% xc,yc,zc  Column vectors with interpolation locations; approximation to
%           be accurate at these points
% 
% Optional:
%     m     The power of the spline; if not specified, 3 will be used
%     d     The power of the polynomial; if it is not specified no
%           polynomials will be included. Requires m to also be set.
%  
% Output parameter 
%    fc     Matrix containing the data values at each collocation point,
%           with columns corresponding to I, d/dx, d/dy, d/dz and each of
%           d2/{dx2,dxy,dy2,dxz,dyz,dz2}, respectively.

if nargin > 7
    m = varargin{1};
    if nargin > 8
        d = varargin{2};
    else
        d = -1;
    end
else
    m = 3;
    d = -1;
end

N = length(xd);
M = length(xc);
fc = zeros(M,10);

for j = 1:M
    w = RBF_FD_PHS_pol_weights_3D (xd,yd,zd,m,d,xc(j),yc(j),zc(j));
    fc(j,:) = w'*fd;
end



end
