%%
clear; close all; clc;

%%
con_nmci = [1, 4, 5, 7, 8, 9, 10, 12, 13, 15, 16, 20, 23, 24];
con_nmci_num = 1:numel(con_nmci);
%rbd_nmci = [1, 6, 7, 8, 9, 10, 13, 14, 16, 18, 22, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 40, 42, 43, 44, 45, 46, 48, 50, 51, 52, 53, 54, 55, 57, 58, 60, 62];
con_nmci = [1, 6, 7, 8, 9, 10, 13, 14, 16, 18, 22, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 40, 42, 43, 44, 45, 46, 48, 50, 51, 52, 53, 54, 55, 57, 58, 60, 62];

a = ground_truth == con_nmci;                   % 바꾸기 rbd_nmci로

b = zeros(size(ground_truth));
c = zeros(size(ground_truth));

for i = 1:numel(con_nmci)
    b(a(:, i)) = i;
end

for i = 1:numel(con_nmci)
    c(a(:, i)) = 1;
end
c = logical(c);

% z = ANT_con_topo_3d(c, :, :, :);
% %not_zero = nnz(c);
% zz = ground_truth(c, :);
% e = zz == con_nmci;
% 
% for i = 1:numel(con_nmci)
%     zz(e(:, i)) = con_nmci_num(i);
% end
% 
% clearvars ANT_con_topo_3d ground_truth
% ANT_con_topo_3d = z;
% ground_truth = zz;

z = ANT_rbd_topo_3d(c, :, :, :);
%not_zero = nnz(c);
zz = ground_truth(c, :);
e = zz == con_nmci;

for i = 1:numel(con_nmci)
    zz(e(:, i)) = con_nmci_num(i);
end


clearvars ANT_rbd_topo_3d ground_truth
ANT_rbd_topo_3d = z;
ground_truth = zz;

%save ANT_con_topo_3d.mat ANT_con_topo_3d ground_truth -v7.3
save ANT_rbd_topo_3d.mat ANT_rbd_topo_3d ground_truth -v7.3