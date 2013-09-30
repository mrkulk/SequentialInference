
%% Calculate Held out likelihood
pc_max_ind = 1e5;
pc_gammaln_by_2 = 1:pc_max_ind;
pc_gammaln_by_2 = gammaln(pc_gammaln_by_2/2);
pc_log_pi = reallog(pi);
pc_log = reallog(1:pc_max_ind);

%inv_Sigma = PF_inv_cov;
%log_det_Sigma = PF_log_det_cov;

FLAG = 0;

new_spike_sortings = zeros(size(spike_sortings));
total_heldout_loglikelihood = 0;
for index=1:10%size(spike_sortings,1)

    %index = map_spike_sorting_index;
    
    if FLAG == 1
       new_spike_sortings(index,:) = ones(1,size(spike_sortings,1));
    else
       new_spike_sortings(index,:) = spike_sortings(index,:);
    end
    
    %PLEASE REMOVE new_spike_sortings = cluster_class(:,1)';



    K = number_of_neurons_in_each_sorting(index);%MAP_number_of_neurons;
    
%     sufficient_stats = {};
%     for kid=1:K %Integrating cluster assigments
%         hits = find(spike_sortings(index,:) ==kid );
%         n = length(hits);
%         data = in_sample_training_data(hits,:);
%         d = size(data,1);
%         mean_Y = mean(data)';
%         
%         mu_n = k_0/(k_0+n)*mu_0 + n/(k_0+n)*mean_Y;
%         k_n = k_0 + n;
%         v_n = v_0 + n;
%         S = sumsqr(data);
%         lambda_n = lambda_0 + S ...
%             + k_0*n/(k_0+n)*(mean_Y-mu_0)*(mean_Y-mu_0)';
%         sufficient_stats{kid} = [mu_n, (lambda_n*(k_n+1))/(k_n*(v_n-d+1))];
%     end

    
    
    heldout_loglikelihood = 0;
    for i=1:size(out_of_sample_training_data, 1)
        %disp(i)
        y = out_of_sample_training_data(i,:)';
        for kid=1:K

            hits = find(new_spike_sortings(index,:) ==kid );
            n = length(hits);
            data = in_sample_training_data(hits,:);
            d = size(data,2);
            m_Y = mean(data,1);
            
            mean_y_vec = ones(size(data));
            mean_y_vec(:,1) = mean_y_vec(:,1).*m_Y(:,1);
            mean_y_vec(:,2) = mean_y_vec(:,2).*m_Y(:,2);

            tmp = data - mean_y_vec;
            SS = zeros(d,d);
            for i=1:size(data,1)
                SS = SS + ( transpose(tmp(i,:))*tmp(i,:) );
            end
    
            [lp ldc ic] = heldout_helper(pc_max_ind,pc_gammaln_by_2,pc_log_pi,pc_log,y,n,m_Y',SS,k_0,mu_0,v_0,lambda_0);  
            heldout_loglikelihood = heldout_loglikelihood + lp;           
        end
    end

    disp('------------');
    disp(index);
    %lp_mvniw(map_spike_sorting(:,1001:8195),inspk(1001:9195,:)', mu_0, k_0,3,lambda_0)
    total_heldout_loglikelihood = total_heldout_loglikelihood + heldout_loglikelihood;
    disp(heldout_loglikelihood);
    %figure(index+10);
    %scatter(in_sample_training_data(:,1),in_sample_training_data(:,2),50,new_spike_sortings(index, :),'.'); hold all;
end

disp('FINAL: ');
disp(total_heldout_loglikelihood/10);
