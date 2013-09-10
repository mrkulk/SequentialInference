%colors = ['r' 'b' 'm' 'g' 'y' 'm' 'y' 'b' 'r' 'g' 'k' 'm' 'y' 'b'];
load times_CSC4
spikes=spikes(1:2:number_of_spikes,:);
cluster_class = cluster_class(1:2:number_of_spikes,:);
cluster_class(:,1) = cluster_class(:,1) + 1;

%map_spike_sorting = spike_sortings(1,:);
%map_spike_sorting = cluster_class(:,1)';


colors={};
DIV=size(unique(map_spike_sorting),2);
elm = [1:255/DIV:255];
elm=elm/255;

colors{1}=[0.8 0 0]; colors{2} = [0 0.8 0]; colors{3} = [0 0 0.8]; colors{4} = [0.8 0.8 0.2];
colors{5} = [0.8 0.4 0.3]; colors{6} = [1 1 1];

colors{7}=[0.2 0 0]; colors{8} = [0 0.2 0]; colors{9} = [0 0 0.2]; colors{10} = [0.2 0.8 0.4];
colors{11} = [0.5 1 0.3]; colors{12} = [1 0 1];

colors{13}=[0.2 0.3 1]; colors{14} = [1 0.2 0]; colors{15} = [0 1 0.6]; colors{16} = [0.9 0.8 0.4];
colors{17} = [0.5 1 1];colors{18} = [0.9 0.8 0];


for i=1:DIV
    colors{i} = colors{i};
end

figure(8)
for i=1:size(map_spike_sorting,2)
   plot(spikes(i,:),'Color', colors{map_spike_sorting(i)});hold all; 
end

% 
% %selective plot
% for ii=0:length(unique(map_spike_sorting))
%     figure(9+ii)
%     for i=1:size(map_spike_sorting,2)
%         if map_spike_sorting(i) == ii
%             plot(spikes(i,:),'Color', colors{map_spike_sorting(i)});hold all; 
%         end
%     end
% end