load('boston.mat');
data = boston.data';
targets = boston.target';

% permute
rng(101);
ind = randperm(size(data,2));
pdata = data(:,ind);
ptargets = targets(:,ind);

% split
tst_ratio = 0.15;
val_ratio = 0.15;

tst_cnt = floor(tst_ratio*size(data,2));
val_cnt = floor(val_ratio*size(data,2));
trn_cnt = size(data,2) - tst_cnt - val_cnt;

RX = pdata(:,1:trn_cnt);
RZ = ptargets(:,1:trn_cnt);
VX = pdata(:, trn_cnt+1 : trn_cnt+val_cnt);
VZ = ptargets(:, trn_cnt+1 : trn_cnt+val_cnt);
TX = pdata(:, trn_cnt+val_cnt+1 : trn_cnt+val_cnt+tst_cnt);
TZ = ptargets(:, trn_cnt+val_cnt+1 : trn_cnt+val_cnt+tst_cnt);

save('boston_split.mat', 'RX', 'RZ', 'VX', 'VZ', 'TX', 'TZ');

