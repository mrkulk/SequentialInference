function [rx] = maxfilter_resample_nonunique(x,w,N)
% function [rx] = multinomial_resample(x,w,N)
%
%  returns N samples from the weighted particle set 
%  x,w with x (DxM) being M, D-dimensional samples and w being a vector of
%  M real weights that sum to one
%
[D, M] = size(x);
rx = zeros(D,N);
rw = zeros(1,N);

[sorted_w, sorted_indx_array] = sort(w, 'descend');

rind=1;
for i=1:N
    sampled_ind = sorted_indx_array(rind);
    rx(:,rind) = x(:,sampled_ind);
    rw(rind) = w(i);
    rind=rind+1;
end
rw = rw./sum(rw);
    
