function [rx] = maxfilter_resample(x,w,N)
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
cnt=1;
record_indx = [];

for i=1:length(sorted_w)
    sampled_ind = sorted_indx_array(i); 
    val = x(:,sampled_ind); %first dim is particle num. second is cluster-id
    val = val(2);
    if isKey(hash, int2str(val)) == 0
        hash(int2str(val)) = 1;
        record_indx(cnt)=sampled_ind;
        cnt=cnt+1;
    end
end


len_rec = length(record_indx);
cnt = 1;
for i = 1:N
    if i <= len_rec
        sampled_ind = record_indx(i);
    else
        sampled_ind = record_indx(1);
        %sampled_ind = sorted_indx_array(cnt);
        cnt=cnt+1;
    end
    rx(:,i) = x(:,sampled_ind);
    rw(i) = w(sampled_ind);
end



% len_rind = length(record_indx);
% pind=1;
% for i = 1:N
%     if pind > len_rind
%         pind=1;
%     end
%     sampled_ind = record_indx(pind);
%     rx(:,i) = x(:,sampled_ind);
%     rw(i) = w(sampled_ind);
%     pind=pind+1;
% end


rw = rw./sum(rw);
