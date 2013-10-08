function [rx,rw] = maxfilter_resample(x,w,N)
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



hash = containers.Map;
for i=1:N
    hash(int2str(i)) = [];
end

for i=1:length(sorted_indx_array)
    sampled_ind = sorted_indx_array(i);
    pid = x(1,sampled_ind);
    tmp = hash(int2str(pid));
    tmp(end+1)=sampled_ind;
    hash(int2str(pid))=tmp;
end


for ii=1:length(hash)
    tmp = hash(int2str(ii));
    offset=ii;
    if ii > 1 && offset <= length(tmp)
        offset=ii-1;
        tmp=tmp([offset+1:end 1:offset]); %left shift by offset    
    end
    hash(int2str(ii))=tmp;
end
    
for i=1:N
   tmp = hash(int2str(i));
   if isempty(tmp) == 0
       sampled_ind = tmp(1);
       %tmp(1)=[];
       tmp=tmp([2:end 1]);
       hash(int2str(i))=tmp;
       rx(:,i) = x(:,sampled_ind);
       rw(i) = w(sampled_ind);
   end
   
   
end


if sum(rw) > 0
    rw = rw./sum(rw);
end

if isnan(rw)
    'notallowed'
end
