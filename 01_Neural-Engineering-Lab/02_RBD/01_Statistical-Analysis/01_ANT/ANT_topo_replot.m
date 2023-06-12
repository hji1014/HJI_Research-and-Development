%% 이전 코드
analysed_results_TN_mean1 = readNPY('C:/Users/Nelab_001/Documents/MATLAB\RBD_ANT/1.LRP\20220204_con-rbd_LRP/LRP_TP_mean_fold10.npy');
analysed_results_TN_mean2 = readNPY('C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\1.LRP\20220204_con-rbd_LRP\LRP_TN_mean_fold10.npy');
analysed_results_TN_mean3 = readNPY('C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\1.LRP\20220204_con-rbd_LRP\LRP_TP_mean_fold9.npy');

figure(1)
for i = 1:14
    subplot(3, 5, i)
    toporeplot(flipud(analysed_results_TN_mean1(:, :, i)), 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', 'jet', 'maplimits', [0, 0.001]);
end


figure(2)
for i = 1:14
    subplot(3, 5, i)
    toporeplot(flipud(analysed_results_TN_mean2(:, :, i)), 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', 'jet', 'maplimits', [0, 0.001]);
end

figure(3)
for i = 1:14
    subplot(3, 5, i)
    toporeplot(flipud(analysed_results_TN_mean3(:, :, i)), 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', 'jet', 'maplimits', [0, 0.001]);
end

% figure(2)
% subplot(1, 4, 1)
% toporeplot(data, 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', 'jet')
% subplot(1, 4, 2)
% toporeplot(data, 'plotrad', 0.5, 'intrad', 1, 'headrad', 1, 'colormap', 'jet')
% subplot(1, 4, 3)
% toporeplot(data, 'plotrad', 1, 'intrad', 0.5, 'headrad', 1, 'colormap', 'jet')
% subplot(1, 4, 4)
% toporeplot(data, 'plotrad', 1, 'intrad', 1, 'headrad', 0.5, 'colormap', 'jet')

%% 10-fold 평균 코드
clc; close all; clear;

% TP(RBD를 RBD로)
LRP_RBD = zeros(67, 67, 14);
for i = 1:10
    address_format1 = 'C:/Users/Nelab_001/Documents/MATLAB/RBD_ANT/1.LRP/20220204_con-rbd_LRP/LRP_TP_mean_fold%d.npy';
    present_address1 = sprintf(address_format1,i);
    LRP_results_foldwise1 = readNPY(present_address1);
    LRP_RBD = LRP_RBD+LRP_results_foldwise1;
end
LRP_RBD = LRP_RBD/i;

% TN (CON을 CON으로)
LRP_CON = zeros(67, 67, 14);
for i = 1:10
    address_format2 = 'C:/Users/Nelab_001/Documents/MATLAB/RBD_ANT/1.LRP/20220204_con-rbd_LRP/LRP_TN_mean_fold%d.npy';
    present_address2 = sprintf(address_format2,i);
    LRP_results_foldwise2 = readNPY(present_address2);
    LRP_CON = LRP_CON+LRP_results_foldwise2;
end
LRP_CON = LRP_CON/i;

figure(1)
for i = 1:14
    subplot(3, 5, i)
    toporeplot(flipud(LRP_RBD(:, :, i)), 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', 'jet', 'maplimits', [0, 0.001]);
    %colorbar;
end

% figure(2)
% for i = 1:14
%     subplot(3, 5, i)
%     toporeplot(LRP_RBD(:, :, i), 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', 'jet', 'maplimits', [0, 0.0012]);
% end

figure(2)
for i = 1:14
    subplot(3, 5, i)
    toporeplot(flipud(LRP_CON(:, :, i)), 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', 'jet', 'maplimits', [0, 0.001]);
end

%% LRP Evaluation

load('C:/Users/Nelab_001/Documents/MATLAB/RBD_ANT/chanlocs_60.mat')
%topo = readNPY('analysed_results_TN_mean.npy');
topo_flip = flipud(LRP_CON);

%%%% 중요 시간대 선정

for i = 1:14
    m = sum(topo_flip(:, :, i), 'all');
    sum_per_frame(i) = m;
end

max_sum = max(sum_per_frame);
max_frame = find(sum_per_frame==max_sum);
threshold = max_sum * 0.8;
important_frame = find(sum_per_frame>threshold);

%%%% 중요 채널 선정
data = topo_flip(:, :, max_frame);

theta = {chanlocs.theta};
radius = {chanlocs.radius};
for i = 1:size(theta,2)
    t(i) = theta{i};
    r(i) = radius{i};
end
t2 = t/180*pi;
r2 = r/(max(r)+0.01);
% polar coordinate를 xy coordinate로 변환
x = r2 .* cos(t2);
y = r2 .* sin(t2);
x2 = 0.5*sqrt(2 + 2*x*sqrt(2) + x.^2 - y.^2) - 0.5*sqrt(2 - 2*x*sqrt(2) + x.^2 - y.^2);
y2 = 0.5*sqrt(2 + 2*y*sqrt(2) - x.^2 + y.^2) - 0.5*sqrt(2 - 2*y*sqrt(2) - x.^2 + y.^2);
% 2D-image meshgrid 생성 (in this case, 67 x 67)
[xq,yq] = meshgrid(linspace(-1,1,67), linspace(-1,1,67));
% flatten data
xq_flat = reshape(xq,1,size(xq,1)*size(xq,2));
yq_flat = reshape(yq,1,size(yq,1)*size(yq,2));
vq_flat = reshape(data,size(data,1)*size(data,2),1);
% flatten data를 xy coordinate에 interpolation
reverse = griddata(xq_flat,yq_flat,double(vq_flat),y2,x2,'v4');

reverse_sort_descend = sort(reverse, 'descend');

figure(3)
% topoplot(reverse, chanlocs, 'emarker2',{find(reverse>mean(reverse)),'o','w',10})
% topoplot(reverse, chanlocs, 'emarker2',{find(reverse>max(reverse)*0.4),'o','w',10})
topoplot(reverse, chanlocs, 'emarker2',{find(reverse>=reverse_sort_descend(6)),'o','w',15},'colormap', 'jet', 'maplimits', [-0.0005, 0.001])
colorbar

%% 연습
clc; close all; clear;

b=0.01:0.01:0.6;
load('C:/Users/Nelab_001/Documents/MATLAB/RBD_ANT/chanlocs_60.mat')
figure(1)
[grid_or_val, plotrad_or_grid] = topoplot(b, chanlocs);
figure(2)
toporeplot(plotrad_or_grid, 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', 'jet')

data = plotrad_or_grid;
theta = {chanlocs.theta};
radius = {chanlocs.radius};
for i = 1:size(theta,2)
    t(i) = theta{i};
    r(i) = radius{i};
end
t2 = t/180*pi;
r2 = r/(max(r)+0.01);
% polar coordinate를 xy coordinate로 변환
x = r2 .* cos(t2);
y = r2 .* sin(t2);
x2 = 0.5*sqrt(2 + 2*x*sqrt(2) + x.^2 - y.^2) - 0.5*sqrt(2 - 2*x*sqrt(2) + x.^2 - y.^2);
y2 = 0.5*sqrt(2 + 2*y*sqrt(2) - x.^2 + y.^2) - 0.5*sqrt(2 - 2*y*sqrt(2) - x.^2 + y.^2);
% 2D-image meshgrid 생성 (in this case, 67 x 67)
[xq,yq] = meshgrid(linspace(-1,1,67), linspace(-1,1,67));
% flatten data
xq_flat = reshape(xq,1,size(xq,1)*size(xq,2));
yq_flat = reshape(yq,1,size(yq,1)*size(yq,2));
vq_flat = reshape(data,size(data,1)*size(data,2),1);
% flatten data를 xy coordinate에 interpolation
reverse = griddata(xq_flat,yq_flat,double(vq_flat),y2,x2,'v4');

figure(3)
topo