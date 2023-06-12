%% Initialize
clc; close all; clear;
%%
load('chan_loc_snuh_60ch.mat') % 분석하고자 하는 전극 채널의 theta, radius 값 로드
theta = {chanlocs.theta};
radius = {chanlocs.radius};
for i = 1:size(theta,2)
    t(i) = theta{i};
    r(i) = radius{i};
end
t2 = t/180*pi;
r2 = r/(max(r)+0.01);
x = r2 .* cos(t2);
y = r2 .* sin(t2);
x2 = 0.5*sqrt(2 + 2*x*sqrt(2) + x.^2 - y.^2) - 0.5*sqrt(2 - 2*x*sqrt(2) + x.^2 - y.^2);
y2 = 0.5*sqrt(2 + 2*y*sqrt(2) - x.^2 + y.^2) - 0.5*sqrt(2 - 2*y*sqrt(2) - x.^2 + y.^2);
[xq,yq] = meshgrid(linspace(-1,1,67), linspace(-1,1,67));

% e.g. griddata(original_x, original_y, data, new_x, new_y, 'interpolation method')
new_data = single(griddata(x2,y2,data,xq,yq,'v4'));
new_data_90 = rot90(new_data);
figure(1)
image(si_N1_90, 'CDataMapping', 'scaled')
colormap('jet')
xticks([])
yticks([])
%colorbar
%saveas(gcf,'a.jpeg')        % 이미지 크롭해서 저장해볼수도.

figure(2)
topoplot(invalid_200_chavg_N1, chanlocs);colorbar;

figure(3)
image(new_data_90(10:57,5:62), 'CDataMapping', 'scaled')
colormap('jet')
xticks([])
yticks([])
%colorbar