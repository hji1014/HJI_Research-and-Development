%% Initialize
clc; close all; clear;

load('result.mat')
chanlocs = importdata('chan_loc_snuh_60ch.mat');
Fs = 400;
N = 400;

tn_200 = linspace(-600, 1400, Fs*2);
tn_1000 = linspace(-1600, 1400, Fs*3);

%% ERP waveforms

figure(1)
for plotid = 1 : 15
    subplot(3, 5, plotid);
    plot(tn_1000, mean(valid_1000_chavg_v_complete(8:12, :), 1), tn_1000, mean(invalid_1000_chavg_v_complete(8:12, :), 1))
    legend('valid 200', 'invalid 200')
    %legend('valid 1000', 'invalid 1000')
    xlim([0 600])
    xticks([-600 0 200 250 600 1400])
    %xlim([-1600 1400])
    %xticks([-1600 0 1400])
    ylim([-3 3])
    xline(0, '--k');
    yline(0, '--k');
    %sgtitle('control 200 valid and control 200 invalid')
    sgtitle('control 1000 valid and control 1000 invalid')
    %sgtitle('RBD 200 valid and RBD 200 invalid')
    %sgtitle('RBD 1000 valid and RBD 1000 invalid')
    
end

plot(tn_200, mean(valid_200_chavg_v_complete(8:12, :), 1), tn_200, mean(invalid_200_chavg_v_complete(8:12, :), 1))
legend('valid 200', 'invalid 200')
%legend('valid 1000', 'invalid 1000')
xlim([-600 1400])
xticks([-600 0 100 200 300 400 500 600 700 1400])
ylim([-3 3])
%xlim([-1600 1400])
%xticks([-1600 0 1400])
xline(0, '--k');
yline(0, '--k');
%sgtitle('control 200 valid and control 200 invalid')
%sgtitle('control 1000 valid and control 1000 invalid')
sgtitle('RBD 200 valid and RBD 200 invalid')
%sgtitle('RBD 1000 valid and RBD 1000 invalid')

a = mean(valid_200_chavg_v_complete(8:12, :), 1);

%% Topographical distribution
%현재 폴더 바꿔서 result 불러오기

% 200-250ms
topo_length = 721:740;

inval = invalid_1000_chavg_v_complete;
val = valid_1000_chavg_v_complete;

% Mean amplitude
inval_amp = mean(inval(:, 721:740), 2);
val_amp = mean(val(:, 721:740), 2);

figure(1)
topoplot(inval_amp, chanlocs, 'maplimits', [-4, 4]);colorbar;

figure(2)
topoplot(val_amp, chanlocs, 'maplimits', [-4, 4]);colorbar;

figure(3)
topoplot(inval_amp-val_amp, chanlocs, 'maplimits', [-1, 1]);colorbar;
