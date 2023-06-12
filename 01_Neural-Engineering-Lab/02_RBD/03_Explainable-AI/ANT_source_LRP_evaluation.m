
% 가장 큰 frame만 그릴 때 (8 frame, 700-800ms)
for i = 1:62
    topo_name = sprintf('D:/ANT_3D_CNN_source_transfer2/LRP_deeptaylor/LRP_TP_sub_%d.npy', i);     %일련의 파일 가져오기 (파일명+숫자.mat일 때)
    topo = readNPY(topo_name);
    topo_mean = squeeze(mean(topo, 1));
    
    % 상위값 찾기 (62846개 중 -> 1%:629 / 5%:3143 / 10%:6285)
    all_value_vector = reshape(topo_mean, 1, []);
    all_value_vector_sort = sort(all_value_vector, 'descend');
    high_value = all_value_vector_sort(3143);
%     
%     % 상위값 미만 0으로
%     for k = 1:62846                     % 62846=67*67*14
%         if topo_mean(k) < high_value
%             topo_mean(k) = 0;
%         end
%     end
    
%     figure(1)
%     toporeplot(topo_mean(:, :, 8), 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', 'jet', 'maplimits', [0, max(all_value_vector_sort)*0.5]);
%     colorbar;

%     figure(3)
%     subplot(7, 10, i)
%     %imagesc(squeeze(flipud(topo_mean(:,:,8))), [0, 1.0e-04])
%     toporeplot(topo_mean(:, :, 8), 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', 'jet', 'maplimits', [0, max(all_value_vector_sort)*0.8]);
%     %toporeplot(topo_mean(:, :, 8), 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', 'jet', 'maplimits', [0, 0.00005]);
    
    % interpolation
    data = topo_mean(:, :, 8);
    load('C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\chanlocs_60.mat')
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
    figure(4)
    subplot(7, 10, i)
    topoplot(reverse, chanlocs, 'maplimits', [0, 0.00007])
    
end

%% TP 전체 피험자(만!!!!) 평균
clc; close all; clear;

topo_all_TP = zeros(67, 67, 14);

load('C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\chanlocs_60.mat')

for i = 1:62
    topo_name = sprintf('D:/ANT_3D_CNN_transfer2/LRP/new/LRP_TP_sub_%d.npy', i);     %일련의 파일 가져오기 (파일명+숫자.mat일 때)
    topo = readNPY(topo_name);
    topo_mean = squeeze(mean(topo, 1));
    
    topo_all_TP = topo_all_TP + topo_mean;
end

topo_all_TP_mean = topo_all_TP/62;

figure(1)
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
    
    subplot(3, 5, i)
    %toporeplot(topo_all_TP_mean(:, :, i), 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', 'jet', 'maplimits', [0, 0.000045]);
    topoplot(reverse, chanlocs, 'maplimits', [0, 0.000046])
    %topoplot(reverse, chanlocs)
    
end