function [rx] = multinomial_resample(x,w,N)
% function [rx] = multinomial_resample(x,w,N)
%
%  returns N samples from the weighted particle set 
%  x,w with x (DxM) being M, D-dimensional samples and w being a vector of
%  M real weights that sum to one
%
[D, M] = size(x);
rx = zeros(D,N);
rw = zeros(1,N);

w = w/sum(w);

rind=1;
for i=1:N
    sampled_ind = mnrnd(1,w);
    sampled_ind = find(sampled_ind == 1);
    rx(:,rind) = x(:,sampled_ind);
    rw(rind) = w(i);
    rind=rind+1;
end
rw = rw./sum(rw);

    
