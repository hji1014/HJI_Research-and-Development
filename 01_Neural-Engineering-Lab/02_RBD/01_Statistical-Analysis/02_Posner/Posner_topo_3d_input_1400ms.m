%%
% Initialize
clc; close all; clear;
%tic

Fs = 200;
tn = linspace(0, 1400, Fs*1.4);

data = importdata('DNN_trgt_2.mat');        % 280 sample -> 1.4s (fs:200)
chanlocs = importdata('chan_loc_snuh_60ch.mat');
con = data.con;
rbd = data.rbd;

con_ch = reshape(con, [280, 60, 9242]);        % reshape
tp_con_ch = permute(con_ch, [2 1 3]);           % 차원 치환
rbd_ch = reshape(rbd, [280, 60, 7929]);
tp_rbd_ch = permute(rbd_ch, [2 1 3]);

clearvars data con rbd con_ch rbd_ch

con_3d_rescale = zeros(9242, 67, 67, 14);
rbd_3d_rescale = zeros(7929, 67, 67, 14);

for i = 1:9242
    for j = 1:14
        range = (j-1)*20+1:j*20;
        avg_100ms = mean(tp_con_ch(:, range, i), 2);
        avg_100ms = rescale(avg_100ms, -1, 1);          % rescale(-1, 1)
        %avg_100ms = normalize(avg_100ms);              % z socore normalize
        [grid_or_val, plotrad_or_grid] = topoplot(avg_100ms, chanlocs,'noplot','on');
        %plotrad_or_grid = fillmissing(plotrad_or_grid, 'constant', 0);
        con_3d_rescale(i, :, :, j) = plotrad_or_grid;
    end
end

for i = 1:7929
    for j = 1:14
        range = (j-1)*20+1:j*20;
        avg_100ms = mean(tp_rbd_ch(:, range, i), 2);
        avg_100ms = rescale(avg_100ms, -1, 1);          % rescale(-1, 1)
        [grid_or_val, plotrad_or_grid] = topoplot(avg_100ms, chanlocs,'noplot','on');
        %plotrad_or_grid = fillmissing(plotrad_or_grid, 'constant', 0);
        rbd_3d_rescale(i, :, :, j) = plotrad_or_grid;
    end
end

toc