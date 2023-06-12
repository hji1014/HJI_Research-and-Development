%% 주의사항

%% Initialize
clc; close all; clear;

%% LPF 20Hz -> Baseline Correction(before stimulus) -> Individual Avg(Ensemble Avg)

%cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\control'
%path = 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\control';   % 'control' path
%path = 'C:\Users\Nelab_001\Documents\MATLAB\RBD_Posner_paradigm\RBD';   % 'RBD' path

%path = 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\1.H_No_cue\Control';      % control No_cue path

% 현재 폴더
path = pwd

fprintf('ERP를 그리는 중입니다...\n')

List = dir(path);
subject_num = length(List);

Ensemble_AVG = zeros(60, 1040, subject_num-2);
%Ensemble_AVG = [];
%EEG_F_BC = [];

for folder_num = 3:subject_num
    %% Path setting
    
    % sprintf 경로 바꾸기
    % control
    foldername = List(folder_num).name;
    %present_valid_200 = sprintf('./control/%s/200_valid.mat', foldername);
    %present_invalid_200 = sprintf('./control/%s/200_invalid.mat', foldername);
    %present_valid_1000 = sprintf('./control/%s/1000_valid.mat', foldername);
    %present_invalid_1000 = sprintf('./control/%s/1000_invalid.mat', foldername);
    % RBD
    %present_valid_200 = sprintf('./RBD/%s/200_valid.mat', foldername);
    %present_invalid_200 = sprintf('./RBD/%s/200_invalid.mat', foldername);
    %present_valid_1000 = sprintf('./RBD/%s/1000_valid.mat', foldername);
    %present_invalid_1000 = sprintf('./RBD/%s/1000_invalid.mat', foldername);
    
    % SOA 200ms
    %valid_200 = importdata(present_valid_200);
    %invalid_200 = importdata(present_invalid_200);
    
    % SOA 1000ms
    %valid_1000 = importdata(present_valid_1000);
    %invalid_1000 = importdata(present_invalid_1000);
    
    % data import
    import_EEG = importdata(foldername);
    EEG = import_EEG.EEG_data;
    
    %% LPF 20Hz hyperparameter setting
    
    Fs = 400;
    N = 400;

    %tn_200 = linspace(-600, 1400, Fs*2);
    %tn_1000 = linspace(-1600, 1400, Fs*3);
    tn_ANT = linspace(-400, 2200, Fs*2.6);      % -400ms~2200ms
    tn_ANT_simple = linspace(-0.4, 2.2, Fs*2.6);    % -0.4s~2.2s

    Nf = 5;         % 5차 필터
    Fp = 20;        % 20Hz LPF
    Ap = 1;         % 통과대역 리플 = 1dB
    As = 60;        % 저지대역 감쇠량 = 60dB

    d = designfilt('lowpassiir','FilterOrder',Nf,'PassbandFrequency',Fp, ...
        'PassbandRipple',Ap,'StopbandAttenuation',As,'SampleRate',Fs);

    % 일반 필터 지연 확인
    % grpdelay(d,N,Fs)

    %xfiltfilt = filtfilt(d, xn);
    %xfiltfilt_BC = xfiltfilt - mean(xfiltfilt(1:160));

    % prestimulus criteria
    % cue -> -600~-200ms / target -> -600~0ms / paper -> -100~0ms
    %cue_onset_200 = 160;
    %target_onest_200 = 240;

    %cue_onset_1000 = 240;
    %target_onset_1000 = 640;
    %paper_onset = 600:640;
    
    baseline_cue = 80:160;          %-200ms~0ms     보통 이걸 사용
    baseline_target = 280:360;      %300ms~500ms

    %% Filtering and Baseline Correction
    
    for i = 1:60        % each ch
        EEG_all = EEG(i, :, :);
        for j = 1:size(EEG_all, 3)       % each trial
            EEG_single_trial = squeeze(EEG_all(:, :, j));
      
            % filtering
            xn = cast(EEG_single_trial, 'double');
            xfiltfilt = filtfilt(d, xn);
      
            % baseline correction
            xfilt_BC = xfiltfilt - mean(xfiltfilt(baseline_cue));
      
            % integration
            EEG(i, :, j) = xfilt_BC;
            %EEG_F_BC(i,:,j) = xfilt_BC;
      
        end
    end

    %% Ensemble AVG
    EEG_Ensemble_AVG = mean(EEG, 3);
    %EEG_Ensemble_AVG = mean(EEG_F_BC, 3);
    Ensemble_AVG(:, :, folder_num-2) = EEG_Ensemble_AVG;
    
    %% Plotting
    %figure(1)
    %for plotid = 1 : 15
    %    subplot(3, 5, plotid);
    %    plot(tn_1000, valid_1000_chavg(plotid, :), tn_1000, invalid_1000_chavg(plotid, :))
    %    legend('valid 1000', 'invalid 1000')
    %    xlim([-1600 1400])
    %    xticks([-1600 0 1400])
    %    xline(0, '--k');
    %end
    
    
end

%% Save file '.mat'
save('D:/ANT_ERP/latest_ERP/rbd_ic', 'Ensemble_AVG')


%% Grand AVG

Grand_AVG = mean(Ensemble_AVG, 3);

%% Save

save('ANT_AVG', 'Ensemble_AVG', 'Grand_AVG')

%% Groupping CH
% a, b, c, d 다 일일히 설정해줘야함

chanlocs = importdata('C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\chanlocs_60.mat');

% 논문 alerting, orienting channel - P3, Pz, P4, O1, O2
%bk = [45 47 49 58 60];                          % P3, Pz, P4, O1, O2
bk = [45 47 49 53, 57, 58 60];                  % P3, Pz, P4, PO7, PO8, O1, O2
%bk = 47;
%bk = 12;

%% Plotting
Fs = 400;
tn_ANT = linspace(-400, 2200, Fs*2.6);      % -400ms~2200ms

%a = importdata('D:/ANT_ERP/latest_ERP/con_cc.mat');
%a = importdata('D:/ANT_ERP/con_cc_ensemble_avg.mat');          % 위랑 똑같은 결과임
a = importdata('C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\ensemble_and_grand_average\2.H_Center_cue\Control\avg.mat');
a = a.data_after_grandavg.avg;
aa = mean(a, 3);
aa = mean(aa(bk, :), 1);
%b = importdata('D:/ANT_ERP/latest_ERP/con_sc.mat');
%b = importdata('D:/ANT_ERP/con_sc_ensemble_avg.mat');
b = importdata('C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\ensemble_and_grand_average\3.H_Spatial_cue\Control\avg.mat');
b = b.data_after_grandavg.avg;
bb = mean(b, 3);
bb = mean(bb(bk, :), 1);
%c = importdata('D:/ANT_ERP/latest_ERP/rbd_cc.mat');
%c = importdata('D:/ANT_ERP/rbd_cc_ensemble_avg.mat');
c = importdata('C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\ensemble_and_grand_average\2.H_Center_cue\RBD\avg.mat');
c = c.data_after_grandavg.avg;
cc = mean(c, 3);
cc = mean(cc(bk, :), 1);
%d = importdata('D:/ANT_ERP/latest_ERP/rbd_sc.mat');
%d = importdata('D:/ANT_ERP/rbd_sc_ensemble_avg.mat');
d = importdata('C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\ensemble_and_grand_average\3.H_Spatial_cue\RBD\avg.mat');
d = d.data_after_grandavg.avg;
dd = mean(d, 3);
dd = mean(dd(bk, :), 1);

figure(1)
plot(tn_ANT, aa, tn_ANT, bb)
legend('cc', 'sc')
title('CON')
figure(2)
plot(tn_ANT, cc, tn_ANT, dd)
legend('cc', 'sc')
title('RBD')

% N1 range : cue 기준(cue start:0 ms) 700-750 ms
N1_range = 441:460;
aaa = a(bk, N1_range, :);
aaa = squeeze(mean(aaa, 1));
aaa = mean(aaa, 1).';

bbb = b(bk, N1_range, :);
bbb = squeeze(mean(bbb, 1));
bbb = mean(bbb, 1).';

ccc = c(bk, N1_range, :);
ccc = squeeze(mean(ccc, 1));
ccc = mean(ccc, 1).';

ddd = d(bk, N1_range, :);
ddd = squeeze(mean(ddd, 1));
ddd = mean(ddd, 1).';