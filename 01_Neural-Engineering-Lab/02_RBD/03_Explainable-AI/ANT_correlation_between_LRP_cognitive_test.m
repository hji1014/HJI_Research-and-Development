%% 
% control -> rbd로만 바꿔주면 됨
% 사용 위치
% 
%% initialization
clear; close all; clc;

%%
cd C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\rbd
path = pwd;
List_condition = dir(path);
condition_num = length(List_condition)-2;
cd C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\rbd\1.No_cue
path = pwd;
List_subject = dir(path);
subject_num = length(List_subject)-2;

con_all = [];
con_all_concat = [];
con_all_trial_mean = [];

Fs = 400;
tn_ANT = linspace(-400, 2200, Fs*2.6);      % -400ms~2200ms
N = 400;
Nf = 5;         % 5차 필터
Fp = 20;        % 20Hz LPF
Ap = 1;         % 통과대역 리플 = 1dB
As = 60;        % 저지대역 감쇠량 = 60dB
d = designfilt('lowpassiir','FilterOrder',Nf,'PassbandFrequency',Fp, ...
    'PassbandRipple',Ap,'StopbandAttenuation',As,'SampleRate',Fs);
baseline_cue = 80:160;
    
for condition = 1:condition_num
    for subject = 1:subject_num
        
        condition_name = List_condition(condition+2).name;
        subjectname = List_subject(subject+2).name;
        data_address = sprintf('C:/Users/Nelab_001/Documents/MATLAB/RBD_ANT/rbd/%s/%s', condition_name, subjectname);
        
        data = importdata(data_address);
        EEG = data.EEG_data;
        
        for i = 1:size(EEG, 1)
            for j = 1:size(EEG, 3)
                trial = EEG(i, :, j);
                trial_double = cast(trial, 'double');
                xfiltfilt = filtfilt(d, trial_double);
                xfilt_BC = xfiltfilt - mean(xfiltfilt(baseline_cue));
                EEG(i, :, j) = xfilt_BC;
            end
        end
        
        con_all{condition, subject} = EEG;
        
    end
end

for k = 1:subject_num
    
    individual_all = cat(3, con_all{:, k});
    con_all_concat{1, k} = individual_all;
    individual_trial_mean = mean(con_all_concat{1, k}, 3);   
    con_all_trial_mean(:, :, k) = individual_trial_mean;
    
end

%%

critical_ch_parietal = [39, 44, 45, 46, 47, 48, 49, 55, 56];
critical_ch_rightfrontal = [14];
t600_900 = 401:520;         % cue 자극 이후 600-900 ms
amplitude_t600_900_parietal = squeeze(mean(con_all_trial_mean(critical_ch_parietal, t600_900, :), 2));           % cue 자극 이후 600-900 ms 평균
amplitude_t600_900_parietal = mean(amplitude_t600_900_parietal, 1).';
amplitude_t600_900_rightfrontal = mean(squeeze(con_all_trial_mean(critical_ch_rightfrontal, t600_900, :)), 1).';           % cue 자극 이후 600-900 ms 평균
