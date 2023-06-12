%% TP 전체 피험자(만!!!!) 평균
clc; close all; clear;

topo_all_TP = zeros(67, 67, 14);

for i = 1:5
    topo_name = sprintf('D:/ANT_3D_CNN_con-rbd+nmci-rbd+mci/LRP/LRP_mci_fold_%d.npy', i);     %일련의 파일 가져오기 (파일명+숫자.mat일 때)
    topo = readNPY(topo_name);
    
    topo_all_TP = topo_all_TP + topo;
end

topo_all_TP_mean = topo_all_TP/5;

% (62846 중 --> 0.1%:63 / 0.5%:315 / 1%:629 / 2%:1257 / 3%:1886 / 5%:3143 / 10%:6285 / 50%:31423)
all_value_vector = reshape(topo_all_TP_mean, 1, []);
all_value_vector_sort = sort(all_value_vector, 'descend');
high_value = all_value_vector_sort(1257);

for i = 1:62846                     % 62846=67*67*14
    if topo_all_TP_mean(i) >= high_value
        topo_all_TP_mean(i) = topo_all_TP_mean(i);
    else
        topo_all_TP_mean(i) = 0;
    end
end

figure(1)
for i = 1:14
    subplot(3, 5, i)
    %toporeplot(topo_all_TP_mean(:, :, i), 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', 'jet', 'maplimits', [0, 0.000045]);
    toporeplot(topo_all_TP_mean(:, :, i), 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', 'jet', 'maplimits',[0, 0.003]);
end

% interpolation
load('C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\chanlocs_60.mat')
topo_all_TP_mean_interpolation = [];
for i = 1:14
    data = topo_all_TP_mean(:, :, i);
    theta = {chanlocs.theta};
    radius = {chanlocs.radius};
    for j = 1:size(theta,2)
        t(j) = theta{j};
        r(j) = radius{j};
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
    reverse = reverse.';
    topo_all_TP_mean_interpolation(:, i) = reverse;    
end


figure(2)
for i = 1:14
    subplot(3, 5, i)
    topoplot(topo_all_TP_mean_interpolation(:, i), chanlocs, 'colormap', 'jet', 'maplimits',[0, 0.003]);
end

% 상위 10개 채널 찾기
a = sort(topo_all_TP_mean_interpolation(:, 8), 'descend');
b = a(10);
c = find(topo_all_TP_mean_interpolation(:, 8) >= b);
figure(3)
topoplot(topo_all_TP_mean_interpolation(:, 8), chanlocs, 'colormap', 'jet', 'maplimits',[0, 0.000055], 'emarker2',{c,'o','w',15})

% interpolation
% data = topo_all_TP_mean(:, :, 8);
% load('C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\chanlocs_60.mat')
% theta = {chanlocs.theta};
% radius = {chanlocs.radius};
% for i = 1:size(theta,2)
%     t(i) = theta{i};
%     r(i) = radius{i};
% end
% t2 = t/180*pi;
% r2 = r/(max(r)+0.01);
% % polar coordinate를 xy coordinate로 변환
% x = r2 .* cos(t2);
% y = r2 .* sin(t2);
% x2 = 0.5*sqrt(2 + 2*x*sqrt(2) + x.^2 - y.^2) - 0.5*sqrt(2 - 2*x*sqrt(2) + x.^2 - y.^2);
% y2 = 0.5*sqrt(2 + 2*y*sqrt(2) - x.^2 + y.^2) - 0.5*sqrt(2 - 2*y*sqrt(2) - x.^2 + y.^2);
% % 2D-image meshgrid 생성 (in this case, 67 x 67)
% [xq,yq] = meshgrid(linspace(-1,1,67), linspace(-1,1,67));
% % flatten data
% xq_flat = reshape(xq,1,size(xq,1)*size(xq,2));
% yq_flat = reshape(yq,1,size(yq,1)*size(yq,2));
% vq_flat = reshape(data,size(data,1)*size(data,2),1);
% % flatten data를 xy coordinate에 interpolation
% reverse = griddata(xq_flat,yq_flat,double(vq_flat),y2,x2,'v4');
% figure(2)
% topoplot(reverse, chanlocs, 'maplimits', [0, 0.00005])


%% TP 전체 피험자(만!!!!) 평균
clc; close all; clear;

topo_all_TP = zeros(67, 67, 14);

for i = 1:5
    topo_name = sprintf('D:/ANT_3D_CNN_con-rbd+nmci-rbd+mci/LRP/LRP_nmci_fold_%d.npy', i);     %일련의 파일 가져오기 (파일명+숫자.mat일 때)
    topo = readNPY(topo_name);
    
    topo_all_TP = topo_all_TP + topo;
end

topo_all_TP_mean = topo_all_TP/5;

% (62846 중 --> 0.1%:63 / 0.5%:315 / 1%:629 / 2%:1257 / 3%:1886 / 5%:3143 / 10%:6285 / 50%:31423)
all_value_vector = reshape(topo_all_TP_mean, 1, []);
all_value_vector_sort = sort(all_value_vector, 'descend');
high_value = all_value_vector_sort(1257);

for i = 1:62846                     % 62846=67*67*14
    if topo_all_TP_mean(i) >= high_value
        topo_all_TP_mean(i) = topo_all_TP_mean(i);
    else
        topo_all_TP_mean(i) = 0;
    end
end

figure(3)
for i = 1:14
    subplot(3, 5, i)
    %toporeplot(topo_all_TP_mean(:, :, i), 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', 'jet', 'maplimits', [0, 0.000045]);
    toporeplot(topo_all_TP_mean(:, :, i), 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', 'jet', 'maplimits',[0, 0.003]);
end

% interpolation
load('C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\chanlocs_60.mat')
topo_all_TP_mean_interpolation = [];
for i = 1:14
    data = topo_all_TP_mean(:, :, i);
    theta = {chanlocs.theta};
    radius = {chanlocs.radius};
    for j = 1:size(theta,2)
        t(j) = theta{j};
        r(j) = radius{j};
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
    reverse = reverse.';
    topo_all_TP_mean_interpolation(:, i) = reverse;    
end


figure(4)
for i = 1:14
    subplot(3, 5, i)
    topoplot(topo_all_TP_mean_interpolation(:, i), chanlocs, 'colormap', 'jet', 'maplimits',[0, 0.003]);
end

% 상위 10개 채널 찾기
a = sort(topo_all_TP_mean_interpolation(:, 8), 'descend');
b = a(10);
c = find(topo_all_TP_mean_interpolation(:, 8) >= b);
figure(3)
topoplot(topo_all_TP_mean_interpolation(:, 8), chanlocs, 'colormap', 'jet', 'maplimits',[0, 0.000055], 'emarker2',{c,'o','w',15})

% interpolation
% data = topo_all_TP_mean(:, :, 8);
% load('C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\chanlocs_60.mat')
% theta = {chanlocs.theta};
% radius = {chanlocs.radius};
% for i = 1:size(theta,2)
%     t(i) = theta{i};
%     r(i) = radius{i};
% end
% t2 = t/180*pi;
% r2 = r/(max(r)+0.01);
% % polar coordinate를 xy coordinate로 변환
% x = r2 .* cos(t2);
% y = r2 .* sin(t2);
% x2 = 0.5*sqrt(2 + 2*x*sqrt(2) + x.^2 - y.^2) - 0.5*sqrt(2 - 2*x*sqrt(2) + x.^2 - y.^2);
% y2 = 0.5*sqrt(2 + 2*y*sqrt(2) - x.^2 + y.^2) - 0.5*sqrt(2 - 2*y*sqrt(2) - x.^2 + y.^2);
% % 2D-image meshgrid 생성 (in this case, 67 x 67)
% [xq,yq] = meshgrid(linspace(-1,1,67), linspace(-1,1,67));
% % flatten data
% xq_flat = reshape(xq,1,size(xq,1)*size(xq,2));
% yq_flat = reshape(yq,1,size(yq,1)*size(yq,2));
% vq_flat = reshape(data,size(data,1)*size(data,2),1);
% % flatten data를 xy coordinate에 interpolation
% reverse = griddata(xq_flat,yq_flat,double(vq_flat),y2,x2,'v4');
% figure(2)
% topoplot(reverse, chanlocs, 'maplimits', [0, 0.00005])

