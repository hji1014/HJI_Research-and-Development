%% 주의사항
% 1. plot() 설정 ->x,y값
% 2. xlim, xticks, legend 설정
% 3. sgtitle 설정
%% Initialize
clc; close all; clear;

%% 
load('result.mat')

Fs = 400;
N = 400;

tn_200 = linspace(-600, 1400, Fs*2);
tn_1000 = linspace(-1600, 1400, Fs*3);

%% Plotting

figure(1)
for plotid = 1 : 15
    subplot(3, 5, plotid);
    plot(tn_200, valid_200_chavg_v_complete(plotid, :), tn_200, invalid_200_chavg_v_complete(plotid, :))
    legend('valid 200', 'invalid 200')
    %legend('valid 1000', 'invalid 1000')
    xlim([-600 1400])
    xticks([-600 0 1400])
    %xlim([-1600 1400])
    %xticks([-1600 0 1400])
    xline(0, '--k');
    %sgtitle('control 200 valid and control 200 invalid')
    %sgtitle('control 1000 valid and control 1000 invalid')
    sgtitle('RBD 200 valid and RBD 200 invalid')
    %sgtitle('RBD 1000 valid and RBD 1000 invalid')
    
end

figure(2)
for plotid = 16 : 30
    subplot(3, 5, plotid-15);
    plot(tn_200, valid_200_chavg_v_complete(plotid, :), tn_200, invalid_200_chavg_v_complete(plotid, :))
    legend('valid 200', 'invalid 200')
    %legend('valid 1000', 'invalid 1000')
    xlim([-600 1400])
    xticks([-600 0 1400])
    %xlim([-1600 1400])
    %xticks([-1600 0 1400])
    xline(0, '--k');
    %sgtitle('control 200 valid and control 200 invalid')
    %sgtitle('control 1000 valid and control 1000 invalid')
    sgtitle('RBD 200 valid and RBD 200 invalid')
    %sgtitle('RBD 1000 valid and RBD 1000 invalid')
end

figure(3)
for plotid = 31 : 45
    subplot(3, 5, plotid-30);
    plot(tn_200, valid_200_chavg_v_complete(plotid, :), tn_200, invalid_200_chavg_v_complete(plotid, :))
    legend('valid 200', 'invalid 200')
    %legend('valid 1000', 'invalid 1000')
    xlim([-600 1400])
    xticks([-600 0 1400])
    %xlim([-1600 1400])
    %xticks([-1600 0 1400])
    xline(0, '--k');
    %sgtitle('control 200 valid and control 200 invalid')
    %sgtitle('control 1000 valid and control 1000 invalid')
    sgtitle('RBD 200 valid and RBD 200 invalid')
    %sgtitle('RBD 1000 valid and RBD 1000 invalid')
end

figure(4)
for plotid = 46 : 60
    subplot(3, 5, plotid-45);
    plot(tn_200, valid_200_chavg_v_complete(plotid, :), tn_200, invalid_200_chavg_v_complete(plotid, :))
    legend('valid 200', 'invalid 200')
    %legend('valid 1000', 'invalid 1000')
    xlim([-600 1400])
    xticks([-600 0 1400])
    %xlim([-1600 1400])
    %xticks([-1600 0 1400])
    xline(0, '--k');
    %sgtitle('control 200 valid and control 200 invalid')
    %sgtitle('control 1000 valid and control 1000 invalid')
    sgtitle('RBD 200 valid and RBD 200 invalid')
    %sgtitle('RBD 1000 valid and RBD 1000 invalid')
end