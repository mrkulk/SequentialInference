mixing_wights = [1/3,1/3,1/3];

%DATASET 1
mu=[0,0;
    1,1;
    2,2;];
std=[0.3,0.3;
    0.3,0.3;
    0.3,0.3;];



NUM_POINTS = 200;
X=zeros(2,NUM_POINTS);

for i=1:NUM_POINTS
   cid = mnrnd(1,mixing_wights);
   cid = find(cid==1);
   X(1,i) = mu(cid,1) + std(cid,1)*randn();
   X(2,i) = mu(cid,2) + std(cid,2)*randn();
end


opts = mkopts_avdp;
opts.get_q_of_z = 1;
results = vdpgm(X, opts);

results.q_of_z

col = [];

tmp={'r','c','g','y', 'b'};

for i=1:length(results.q_of_z)
    cid = mnrnd(1,results.q_of_z(i,:));
    cid = find(cid==1);
    col(i,:) = tmp{cid};
end

scatter(X(1,:), X(2,:),36,col,'.');

