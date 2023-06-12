%%
% Initialize
clc; close all; clear;
tic

Fs = 200;
tn = linspace(0, 1400, Fs*1.4);             % N1 -> 41:50 (200ms-250ms)

data = importdata('DNN_trgt_2.mat');        % 280 sample -> 1.4s (fs:200)
chanlocs = importdata('chan_loc_snuh_60ch.mat');
con = data.con;
rbd = data.rbd;

con_ch = reshape(con, [280, 60, 9242]);        % reshape
tp_con_ch = permute(con_ch, [2 1 3]);           % 차원 치환
rbd_ch = reshape(rbd, [280, 60, 7929]);
tp_rbd_ch = permute(rbd_ch, [2 1 3]);

clearvars data con rbd con_ch rbd_ch

con_2d_7 = zeros(9242, 67, 67);
rbd_2d_7 = zeros(7929, 67, 67);

%% filter

Nf = 3;         % 7차 필터
Fp = 20;        % 20Hz LPF
Ap = 1;         % 통과대역 리플 = 1dB
As = 60;        % 저지대역 감쇠량 = 60dB

d = designfilt('lowpassiir','FilterOrder',Nf,'PassbandFrequency',Fp, ...
    'PassbandRipple',Ap,'StopbandAttenuation',As,'SampleRate',Fs);

%% 1. filtering 기반
for i = 1:9242
    for j = 1:60
        %tp_con_ch(j, :, i) = filtfilt(d, tp_con_ch(j, :, i));
        filtering = filtfilt(d, tp_con_ch(j, :, i));
        bc = filtering - mean(filtering(:, 1:40), 2);
        tp_con_ch(j, :, i) = bc;
    end
end
for i = 1:9242
    N1 = mean(tp_con_ch(:, 41:50, i), 2);
    %N1 = normalize(N1);              % mean=0, var=1 (z-score normalization)
    %N1 = rescale(N1, -1, 1);          % rescale(-1, 1)
    [grid_or_val, plotrad_or_grid] = topoplot(N1, chanlocs,'noplot','on');
    con_2d_7(i, :, :) = plotrad_or_grid;
end

for i = 1:7929
    for j = 1:60
        %tp_rbd_ch(j, :, i) = filtfilt(d, tp_rbd_ch(j, :, i));
        filtering = filtfilt(d, tp_rbd_ch(j, :, i));
        bc = filtering - mean(filtering(:, 1:40), 2);
        tp_rbd_ch(j, :, i) = bc;
    end
end
for i = 1:7929
    N1 = mean(tp_rbd_ch(:, 41:50, i), 2);
    %N1 = normalize(N1);              % mean=0, var=1 (z-score normalization)
    %N1 = rescale(N1, -1, 1);          % rescale(-1, 1)
    [grid_or_val, plotrad_or_grid] = topoplot(N1, chanlocs,'noplot','on');
    rbd_2d_7(i, :, :) = plotrad_or_grid;
end

toc
%% 2. filter + si 기반 square interpolation
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

for i = 1:9242
    %for j = 1:60
    %    tp_con_ch(j, :, i) = filtfilt(d, tp_con_ch(j, :, i));        
    %end
    N1 = mean(tp_con_ch(:, 41:50, i), 2);
    si_N1 = single(griddata(x2, y2, N1, xq, yq, 'v4'));
    si_N1_90 = rot90(si_N1);
    con_2d_6(i, :, :) = si_N1_90;
end

for i = 1:7929
    %for j = 1:60
    %    tp_rbd_ch(j, :, i) = filtfilt(d, tp_rbd_ch(j, :, i));        
    %end
    N1 = mean(tp_rbd_ch(:, 41:50, i), 2);
    si_N1 = single(griddata(x2, y2, N1, xq, yq, 'v4'));
    si_N1_90 = rot90(si_N1);
    rbd_2d_6(i, :, :) = si_N1_90;
end

%% 2. raw EEG normalize 기반(all points)

%%
toc