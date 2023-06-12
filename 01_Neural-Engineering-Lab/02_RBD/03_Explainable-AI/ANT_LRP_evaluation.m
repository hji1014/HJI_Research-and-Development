clc; close all; clear;

%% 흰색~빨간색 으로만 그리기
    a = 0:0.001:1;
    a = flip(a).';
    b = 0:0.001:1;
    b = flip(b).';
    c = ones(1001, 1);
    map = [c b a];
%     figure(i)
%     for j = 1:14
%         subplot(3, 5, j)
%         toporeplot(topo_mean(:, :, j), 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', map, 'maplimits', [0, max(all_value_vector_sort)*0.5]);
%         colorbar();
%     end

%% TP 전체 피험자(만!!!!) 평균
clc; close all; clear;

topo_all_TP = zeros(67, 67, 14);

for i = 1:62
    topo_name = sprintf('D:/ANT_3D_CNN_transfer2/LRP/pre/LRP_TP_sub_%d.npy', i);     %일련의 파일 가져오기 (파일명+숫자.mat일 때)
    topo = readNPY(topo_name);
    topo_mean = squeeze(mean(topo, 1));
    
    topo_all_TP = topo_all_TP + topo_mean;
end

topo_all_TP_mean = topo_all_TP/62;

% (62846 중 --> 0.1%:63 / 0.5%:315 / 1%:629 / 3%:1886 / 5%:3143 / 10%:6285 / 50%:31423)
all_value_vector = reshape(topo_all_TP_mean, 1, []);
all_value_vector_sort = sort(all_value_vector, 'descend');
high_value = all_value_vector_sort(1886);

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
    toporeplot(topo_all_TP_mean(:, :, i), 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', 'jet', 'maplimits',[0, 0.00007]);
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
    topoplot(topo_all_TP_mean_interpolation(:, i), chanlocs, 'colormap', 'jet', 'maplimits',[0, 0.000055]);
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


%% 개인별 TP

% 62명 다 그릴 때
for i = 1:16
    topo_name = sprintf('D:/ANT_3D_CNN_transfer/LRP_deeptaylor/LRP_TP_sub_%d.npy', i);     %일련의 파일 가져오기 (파일명+숫자.mat일 때)
    topo = readNPY(topo_name);
    topo_mean = squeeze(mean(topo, 1));
    
    % 상위값 찾기 (62846개 중 -> 1%:629 / 5%:3143 / 10%:6285)
    all_value_vector = reshape(topo_mean, 1, []);
    all_value_vector_sort = sort(all_value_vector, 'descend');
    high_value = all_value_vector_sort(62846);
    
    % 상위값 미만 0으로
    for k = 1:62846                     % 62846=67*67*14
        if topo_mean(k) >= high_value
            topo_mean(k) = topo_mean(k);
        else
            topo_mean(k) = 0;
        end
    end

    figure(i)
    for j = 1:14
       % subplot(3, 5, j)
        %toporeplot(topo_mean(:, :, j), 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'maplimits', [0, max(all_value_vector_sort)*0.5]);
        toporeplot(topo_mean(:, :, j), 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', 'jet');
        %toporeplot(topo_mean(:, :, j), 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', 'jet', 'maplimits', [0, 0.001]);
        colorbar();
    end
end

% 가장 큰 frame만 그릴 때 (8 frame, 700-800ms)
for i = 1:62
    topo_name = sprintf('D:/ANT_3D_CNN_transfer/LRP_deeptaylor/LRP_TP_sub_%d.npy', i);     %일련의 파일 가져오기 (파일명+숫자.mat일 때)
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

    figure(3)
    subplot(7, 10, i)
    %imagesc(squeeze(flipud(topo_mean(:,:,8))), [0, 1.0e-04])
    toporeplot(topo_mean(:, :, 8), 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', 'jet', 'maplimits', [0, max(all_value_vector_sort)*0.8]);
    %toporeplot(topo_mean(:, :, 8), 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', 'jet', 'maplimits', [0, 0.00005]);
    
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
    topoplot(reverse, chanlocs, 'maplimits', [0, 0.00005])
    
end

%% pretrained classifier TP 전체 평균


%% TP 전체 피험자 평균
% topo_all_TP = zeros(67, 67, 14);
% 
% for i = 1:62
%     topo_name = sprintf('D:/ANT_3D_CNN_transfer/LRP/LRP_TP_sub_%d.npy', i);     %일련의 파일 가져오기 (파일명+숫자.mat일 때)
%     topo = readNPY(topo_name);
%     topo_mean = squeeze(mean(topo, 1));
%     
%     topo_all_TP = topo_all_TP + topo_mean;
% end
% 
% topo_all_TP_mean = topo_all_TP/62;
% 
% %% FN 전체 피험자 평균
% FN_sub = [1, 5, 10, 12, 13, 15, 16, 18, 19, 23, 25, 27, 29, 30, 32, 35, 38, 40, 48, 59];
% topo_all_FN = zeros(67, 67, 14);
% 
% for i = 1:numel(FN_sub)
%     FN_sub_order = FN_sub(i);
%     topo_name = sprintf('D:/ANT_3D_CNN_transfer/LRP/LRP_FN_sub_%d.npy', FN_sub_order);     %일련의 파일 가져오기 (파일명+숫자.mat일 때)
%     topo = readNPY(topo_name);
%     topo_mean = squeeze(mean(topo, 1));
%     
%     topo_all_FN = topo_all_FN + topo_mean;
% end
% 
% topo_all_FN_mean = topo_all_FN/20;
% 
% %% TP와 FN의 상위 n% 값 찾기 (125692개 중 --> 0.1%:126 / 0.5%:629 / 1%:1257 / 5%:6285 / 10%:12570 / 50%:)
% all_value = [topo_all_TP_mean topo_all_FN_mean];
% all_value_vector = reshape(all_value, 1, []);
% all_value_vector_sort = sort(all_value_vector, 'descend');
% high_value = all_value_vector_sort(62846);
% 
% %% thresholding (상위 n% 값 이내 값은 0으로 변환)
% for i = 1:62846                     % 62846=67*67*14
%     if topo_all_TP_mean(i) >= high_value
%         topo_all_TP_mean(i) = topo_all_TP_mean(i);
%     else
%         topo_all_TP_mean(i) = 0;
%     end
% end
% 
% for i = 1:62846                     % 62846=67*67*14
%     if topo_all_FN_mean(i) >= high_value
%         topo_all_FN_mean(i) = topo_all_FN_mean(i);
%     else
%         topo_all_FN_mean(i) = 0;
%     end
% end
% 
% %% plotting
% figure(1)
% for i = 1:14
%     subplot(3, 5, i)
%     toporeplot(topo_all_TP_mean(:, :, i), 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', 'jet', 'maplimits', [0, 0.0001]);
%     
% end
% 
% figure(2)
% for i = 1:14
%     subplot(3, 5, i)
%     toporeplot(topo_all_FN_mean(:, :, i), 'plotrad', 1, 'intrad', 1, 'headrad', 1, 'colormap', 'jet');
% end

%% time 선정, ch 선정, plot (구 방법)

for i = 1:14
    m = sum(topo_flip(:, :, i), 'all');
    sum_per_frame(i) = m;
end

max_sum = max(sum_per_frame);
max_frame = find(sum_per_frame==max_sum);
threshold = max_sum * 0.8;
important_frame = find(sum_per_frame>threshold);

%

data = topo_flip(:, :, max_frame);

load('C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\chanlocs_60.mat')
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

figure(2)
% topoplot(reverse, chanlocs, 'emarker2',{find(reverse>mean(reverse)),'o','w',10})
% topoplot(reverse, chanlocs, 'emarker2',{find(reverse>max(reverse)*0.4),'o','w',10})
topoplot(reverse, chanlocs, 'emarker2',{find(reverse>=reverse_sort_descend(6)),'o','w',15})

%

% % 9 region
% r1 = topo_flip(1:23, 1:23, max_frame);
% r2 = topo_flip(1:23, 24:45, max_frame);
% r3 = topo_flip(1:23, 46:67, max_frame);
% r4 = topo_flip(24:45, 1:23, max_frame);
% r5 = topo_flip(24:45, 24:45, max_frame);
% r6 = topo_flip(24:45, 46:67, max_frame);
% r7 = topo_flip(46:67, 1:23, max_frame);
% r8 = topo_flip(46:67, 24:45, max_frame);
% r9 = topo_flip(46:67, 46:67, max_frame);
% 
% % 0이 아닌 인덱스 개수
% r1_num_except_zero = numel(find(r1~=0));
% r2_num_except_zero = numel(find(r2~=0));
% r3_num_except_zero = numel(find(r3~=0));
% r4_num_except_zero = numel(find(r4~=0));
% r5_num_except_zero = numel(find(r5~=0));
% r6_num_except_zero = numel(find(r6~=0));
% r7_num_except_zero = numel(find(r7~=0));
% r8_num_except_zero = numel(find(r8~=0));
% r9_num_except_zero = numel(find(r9~=0));
% 
% % 0을 제외하여 구한 평균
% r1_mean_except_zero = sum(r1, 'all')/(numel(r1)-r1_num_except_zero);
% r2_mean_except_zero = sum(r2, 'all')/(numel(r2)-r2_num_except_zero);
% r3_mean_except_zero = sum(r3, 'all')/(numel(r3)-r3_num_except_zero);
% r4_mean_except_zero = sum(r4, 'all')/(numel(r4)-r4_num_except_zero);
% r5_mean_except_zero = sum(r5, 'all')/(numel(r5)-r5_num_except_zero);
% r6_mean_except_zero = sum(r6, 'all')/(numel(r6)-r6_num_except_zero);
% r7_mean_except_zero = sum(r7, 'all')/(numel(r7)-r7_num_except_zero);
% r8_mean_except_zero = sum(r8, 'all')/(numel(r8)-r8_num_except_zero);
% r9_mean_except_zero = sum(r9, 'all')/(numel(r9)-r9_num_except_zero);
% 
% % 0을 제외하여 구한 평균보다 큰 인덱스 개수
% r1_over_num = numel(find(r1>r1_mean_except_zero));
% r2_over_num = numel(find(r2>r2_mean_except_zero));
% r3_over_num = numel(find(r3>r3_mean_except_zero));
% r4_over_num = numel(find(r4>r4_mean_except_zero));
% r5_over_num = numel(find(r5>r5_mean_except_zero));
% r6_over_num = numel(find(r6>r6_mean_except_zero));
% r7_over_num = numel(find(r7>r7_mean_except_zero));
% r8_over_num = numel(find(r8>r8_mean_except_zero));
% r9_over_num = numel(find(r9>r9_mean_except_zero));
% 
% % important region
% r1_percentage = (r1_over_num/r1_num_except_zero)*100;
% r2_percentage = (r2_over_num/r2_num_except_zero)*100;
% r3_percentage = (r3_over_num/r3_num_except_zero)*100;
% r4_percentage = (r4_over_num/r4_num_except_zero)*100;
% r5_percentage = (r5_over_num/r5_num_except_zero)*100;
% r6_percentage = (r6_over_num/r6_num_except_zero)*100;
% r7_percentage = (r7_over_num/r7_num_except_zero)*100;
% r8_percentage = (r8_over_num/r8_num_except_zero)*100;
% r9_percentage = (r9_over_num/r9_num_except_zero)*100;
% 
% important_region = [r1_percentage r2_percentage r3_percentage r4_percentage r5_percentage r6_percentage r7_percentage r8_percentage r9_percentage];

