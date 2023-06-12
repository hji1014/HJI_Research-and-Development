%% ANT ERP plotting 구버전
clc; close all; clear;

Fs = 400;
tn = linspace(-400, 2200, Fs*2.6).';

% No_Cue, Center_Cue, Spatial_Cue
figure(1)

subplot(1, 2, 1);
plot(tn(80:640), mean(N_C([53, 57], 80:640), 1), 'k', 'LineWidth',2)
hold on
plot(tn(80:640), mean(C_C([53, 57], 80:640), 1), 'r', 'LineWidth',2)
plot(tn(80:640), mean(S_C([53, 57], 80:640), 1), 'b', 'LineWidth',2)
legend('No cue', 'Center cue', 'Spatial cue')
hold off

subplot(1, 2, 2);
plot(tn(80:640), mean(N_R([53, 57], 80:640), 1), 'k', 'LineWidth',2)
hold on
plot(tn(80:640), mean(C_R([53, 57], 80:640), 1), 'r', 'LineWidth',2)
plot(tn(80:640), mean(S_R([53, 57], 80:640), 1), 'b', 'LineWidth',2)
legend('No cue', 'Center cue', 'Spatial cue')
hold off

% Congruent, Incongruent
figure(2)

subplot(1, 2, 1);
plot(tn(80:640), mean(CG_C([47], 80:640), 1), 'k', 'LineWidth',2) % 12 30 47
hold on
plot(tn(80:640), mean(ICG_C([47], 80:640), 1), 'r', 'LineWidth',2)
legend('Congruent', 'Incongruent')
hold off

subplot(1, 2, 2);
plot(tn(80:640), mean(CG_R([47], 80:640), 1), 'k', 'LineWidth',2)
hold on
plot(tn(80:640), mean(ICG_R([47], 80:640), 1), 'r', 'LineWidth',2)
legend('Congruent', 'Incongruent')
hold off

% Congruent, Incongruent
figure(2)
Fz=12, Cz=30, Pz=47
subplot(2, 1, 1);
plot(tn, mean(CG_C(Pz, :), 1), 'k', 'LineWidth',2) % 12 30 47
hold on
plot(tn, mean(ICG_C(Pz, :), 1), 'r', 'LineWidth',2)
legend('Congruent', 'Incongruent')
hold off

subplot(2, 1, 2);
plot(tn, mean(CG_R(Pz, :), 1), 'k', 'LineWidth',2)
hold on
plot(tn, mean(ICG_R(Pz, :), 1), 'r', 'LineWidth',2)
legend('Congruent', 'Incongruent')
hold off

%% ANT ERP plotting 신버전
clear all; close all; clc;
Fs = 400;
tn = linspace(-400, 2200, Fs*2.6).';
tn_analysis = linspace(-300, 1500, Fs*1.8).';        % 1040기준->40:760 /
tn_target = linspace(-800, 1000, Fs*1.8).';
% tn_analysis = linspace(-300, 1300, Fs*1.6).';        % 1040기준->40:480
samples_analysis = 41:760; 
%samples_analysis = 41:680;

% P3 : 1000ms~1300ms(target 기준 500ms-800ms)
tn_P3 =  linspace(500, 800, Fs*0.3).';  % target 기준 500ms-800ms
P3_analysis = 561:680;  % conflict
P3_analysis_500_to_600 = 561:600;
P3_analysis_600_to_700 = 601:640;
P3_analysis_700_to_800 = 641:680;

%N1_analysis = 441:501;         %orienting! 700-750 / 650 750 / 700 720 / 700 850
N1_analysis= 441:460;           %alerting! 180-250 680-750 / 650-750 / 220 250 720 750 / 250 260 750 760

% ****** <orienting effect(center cue - spatial cue)> ******
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT'
load('chanlocs_60.mat')
Fz = 12;
Pz = 47;
orienting_alerting_ch = [45 47 49 53 57 58 60];

%con
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\2.H_Center_cue\Control'
load('avg.mat')
empty = [];
for i = 1:size(all_subject, 2)
    empty(i, :, :) = all_subject{i}.avg;
end
a = mean(data_after_grandavg.avg(orienting_alerting_ch, :), 1);
a_N1 = mean(squeeze(mean(empty(:, orienting_alerting_ch, N1_analysis), 2)), 2);
a_N1_mean_amplitude = mean(empty(:, :, N1_analysis), 3);
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\3.H_Spatial_cue\Control'
load('avg.mat')
empty = [];
for i = 1:size(all_subject, 2)
    empty(i, :, :) = all_subject{i}.avg;
end
b = mean(data_after_grandavg.avg(orienting_alerting_ch, :), 1);
b_N1 = mean(squeeze(mean(empty(:, orienting_alerting_ch, N1_analysis), 2)), 2);
b_N1_mean_amplitude = mean(empty(:, :, N1_analysis), 3);

[h_CON, p_CON, ci_CON, stats_CON] = ttest(a_N1, b_N1, 'Alpha', 0.05);                        %%%

%rbd
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\2.H_Center_cue\RBD'
load('avg.mat')
empty = [];
for i = 1:size(all_subject, 2)
    empty(i, :, :) = all_subject{i}.avg;
end
c = mean(data_after_grandavg.avg(orienting_alerting_ch, :), 1);
c_N1 = mean(squeeze(mean(empty(:, orienting_alerting_ch, N1_analysis), 2)), 2);
c_N1_mean_amplitude = mean(empty(:, :, N1_analysis), 3);
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\3.H_Spatial_cue\RBD'
load('avg.mat')
empty = [];
for i = 1:size(all_subject, 2)
    empty(i, :, :) = all_subject{i}.avg;
end
d = mean(data_after_grandavg.avg(orienting_alerting_ch, :), 1);
d_N1 = mean(squeeze(mean(empty(:, orienting_alerting_ch, N1_analysis), 2)), 2);
d_N1_mean_amplitude = mean(empty(:, :, N1_analysis), 3);

[h_RBD, p_RBD, ci_RBD, stats_RBD] = ttest(c_N1, d_N1, 'Alpha', 0.05);

figure; % con center - spatial N1 mean amplitude
subplot(2, 1, 1)
%topoplot(mean(a_N1_mean_amplitude, 1),chanlocs, 'maplimits',[-0.4,0.4]);colorbar;
topoplot(mean(a_N1_mean_amplitude, 1),chanlocs, 'maplimits',[-2, 2]);colorbar;
subplot(2, 1, 2)
topoplot(mean(b_N1_mean_amplitude, 1),chanlocs, 'maplimits',[-2, 2]);colorbar;

figure; % rbd center - spatial N1 mean amplitude
subplot(2, 1, 1)
%topoplot(mean(a_N1_mean_amplitude, 1),chanlocs, 'maplimits',[-0.4,0.4]);colorbar;
topoplot(mean(c_N1_mean_amplitude, 1),chanlocs, 'maplimits',[-2, 2]);colorbar;
subplot(2, 1, 2)
topoplot(mean(d_N1_mean_amplitude, 1),chanlocs, 'maplimits',[-2, 2]);colorbar;

figure;
subplot(1, 2, 1)
plot(tn_target, a(samples_analysis), 'r', 'LineWidth',2)
xlim([-800 1000])
ylim([-2 1.5])
hold on
plot(tn_target, b(samples_analysis), 'b', 'LineWidth',2)
xlim([-800 1000])
ylim([-2 1.5])
xline(0, '-', {'target', 'onset'})
xline(-500, ':', {'cue', 'onset'})
yline(0)
legend('Center cue', 'Spatial cue')
xlabel('time[ms]')
ylabel('amplitude[uV]')
hold off
subplot(1, 2, 2)
plot(tn_target, c(samples_analysis), 'r', 'LineWidth',2)
xlim([-800 1000])
ylim([-2 1.5])
hold on
plot(tn_target, d(samples_analysis), 'b', 'LineWidth',2)
xlim([-800 1000])
ylim([-2 1.5])
xline(0, '-', {'target', 'onset'})
xline(-500, ':', {'cue', 'onset'})
yline(0)
legend('Center cue', 'Spatial cue')
xlabel('time[ms]')
ylabel('amplitude[uV]')
hold off

z = mean(a_N1);
zz = mean(b_N1);
zzz = mean(c_N1);
zzzz = mean(d_N1);

% ****** <alerting effect(center cue - no cue)> ******
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT'
load('chanlocs_60.mat')
Fz = 12;
Pz = 47;
orienting_alerting_ch = [45 47 49 53 57 58 60];

%con
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\2.H_Center_cue\Control'
load('avg.mat')
empty = [];
for i = 1:size(all_subject, 2)
    empty(i, :, :) = all_subject{i}.avg;
end
a = mean(data_after_grandavg.avg(orienting_alerting_ch, :), 1);
a_N1 = mean(squeeze(mean(empty(:, orienting_alerting_ch, N1_analysis), 2)), 2);
a_N1_mean_amplitude = mean(empty(:, :, N1_analysis), 3);
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\1.H_No_cue\Control'
load('avg.mat')
empty = [];
for i = 1:size(all_subject, 2)
    empty(i, :, :) = all_subject{i}.avg;
end
b = mean(data_after_grandavg.avg(orienting_alerting_ch, :), 1);
b_N1 = mean(squeeze(mean(empty(:, orienting_alerting_ch, N1_analysis), 2)), 2);
b_N1_mean_amplitude = mean(empty(:, :, N1_analysis), 3);

[h_CON, p_CON, ci_CON, stats_CON] = ttest(a_N1, b_N1, 'Alpha', 0.05);                        %%%

%rbd
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\2.H_Center_cue\RBD'
load('avg.mat')
empty = [];
for i = 1:size(all_subject, 2)
    empty(i, :, :) = all_subject{i}.avg;
end
c = mean(data_after_grandavg.avg(orienting_alerting_ch, :), 1);
c_N1 = mean(squeeze(mean(empty(:, orienting_alerting_ch, N1_analysis), 2)), 2);
c_N1_mean_amplitude = mean(empty(:, :, N1_analysis), 3);
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\1.H_No_cue\RBD'
load('avg.mat')
empty = [];
for i = 1:size(all_subject, 2)
    empty(i, :, :) = all_subject{i}.avg;
end
d = mean(data_after_grandavg.avg(orienting_alerting_ch, :), 1);
d_N1 = mean(squeeze(mean(empty(:, orienting_alerting_ch, N1_analysis), 2)), 2);
d_N1_mean_amplitude = mean(empty(:, :, N1_analysis), 3);

[h_RBD, p_RBD, ci_RBD, stats_RBD] = ttest(c_N1, d_N1, 'Alpha', 0.05);

figure; % con center - spatial N1 mean amplitude
subplot(2, 1, 1)
%topoplot(mean(a_N1_mean_amplitude, 1),chanlocs, 'maplimits',[-0.4,0.4]);colorbar;
topoplot(mean(a_N1_mean_amplitude, 1),chanlocs, 'maplimits',[-2, 2]);colorbar;
subplot(2, 1, 2)
topoplot(mean(b_N1_mean_amplitude, 1),chanlocs, 'maplimits',[-2, 2]);colorbar;

figure; % rbd center - spatial N1 mean amplitude
subplot(2, 1, 1)
%topoplot(mean(a_N1_mean_amplitude, 1),chanlocs, 'maplimits',[-0.4,0.4]);colorbar;
topoplot(mean(c_N1_mean_amplitude, 1),chanlocs, 'maplimits',[-2, 2]);colorbar;
subplot(2, 1, 2)
topoplot(mean(d_N1_mean_amplitude, 1),chanlocs, 'maplimits',[-2, 2]);colorbar;

figure;
subplot(1, 2, 1)
plot(tn_target, a(samples_analysis), 'r', 'LineWidth',2)
xlim([-800 1000])
ylim([-2 1.5])
hold on
plot(tn_target, b(samples_analysis), 'k', 'LineWidth',2)
xlim([-800 1000])
ylim([-2 1.5])
xline(0, '-', {'target', 'onset'})
xline(-500, ':', {'cue', 'onset'})
yline(0)
legend('Center cue', 'No cue')
xlabel('time[ms]')
ylabel('amplitude[uV]')
hold off
subplot(1, 2, 2)
plot(tn_target, c(samples_analysis), 'r', 'LineWidth',2)
xlim([-800 1000])
ylim([-2 1.5])
hold on
plot(tn_target, d(samples_analysis), 'k', 'LineWidth',2)
xlim([-800 1000])
ylim([-2 1.5])
xline(0, '-', {'target', 'onset'})
xline(-500, ':', {'cue', 'onset'})
yline(0)
legend('Center cue', 'No cue')
xlabel('time[ms]')
ylabel('amplitude[uV]')
hold off

z = mean(a_N1);
zz = mean(b_N1);
zzz = mean(c_N1);
zzzz = mean(d_N1);

% ****** <executive control effect(incongruent-congruent)> ******

cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT'
load('chanlocs_60.mat')
Fz = 12;
Pz = 47;
%con
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\5.H_Incongruent\Control'
load('avg.mat')
empty = [];
for i = 1:size(all_subject, 2)
    empty(i, :, :) = all_subject{i}.avg;
end
a = data_after_grandavg.avg(Fz, :);
a_P3 = mean(squeeze(empty(:, Fz, P3_analysis)), 2);
a_P3_mean_amplitude1 = mean(empty(:, :, P3_analysis_500_to_600), 3);
a_P3_mean_amplitude2 = mean(empty(:, :, P3_analysis_600_to_700), 3);
a_P3_mean_amplitude3 = mean(empty(:, :, P3_analysis_700_to_800), 3);
aa = data_after_grandavg.avg(Pz, :);
aa_P3 = mean(squeeze(empty(:, Pz, P3_analysis_600_to_700)), 2);
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\4.H_Congruent\Control'
load('avg.mat')
empty = [];
for i = 1:size(all_subject, 2)
    empty(i, :, :) = all_subject{i}.avg;
end
b = data_after_grandavg.avg(Fz, :);
b_P3 = mean(squeeze(empty(:, Fz, P3_analysis)), 2);
b_P3_mean_amplitude1 = mean(empty(:, :, P3_analysis_500_to_600), 3);
b_P3_mean_amplitude2 = mean(empty(:, :, P3_analysis_600_to_700), 3);
b_P3_mean_amplitude3 = mean(empty(:, :, P3_analysis_700_to_800), 3);
bb = data_after_grandavg.avg(Pz, :);
bb_P3 = mean(squeeze(empty(:, Pz, P3_analysis_600_to_700)), 2);
bb_P3_mean_amplitude = mean(empty(:, :, P3_analysis_600_to_700), 3);

[h_CON_Fz, p_CON_Fz, ci_CON_Fz, stats_CON_Fz] = ttest(a_P3, b_P3, 'Alpha', 0.01);
[h_CON_Pz, p_CON_Pz, ci_CON_Pz, stats_CON_Pz] = ttest(aa_P3, bb_P3, 'Alpha', 0.01);

%rbd
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\5.H_Incongruent\RBD'
load('avg.mat')
empty = [];
for i = 1:size(all_subject, 2)
    empty(i, :, :) = all_subject{i}.avg;
end
c = data_after_grandavg.avg(Fz, :);
c_P3 = mean(squeeze(empty(:, Fz, P3_analysis)), 2);
c_P3_mean_amplitude1 = mean(empty(:, :, P3_analysis_500_to_600), 3);
c_P3_mean_amplitude2 = mean(empty(:, :, P3_analysis_600_to_700), 3);
c_P3_mean_amplitude3 = mean(empty(:, :, P3_analysis_700_to_800), 3);
cc = data_after_grandavg.avg(Pz, :);
cc_P3 = mean(squeeze(empty(:, Pz, P3_analysis)), 2);
cc_P3_mean_amplitude = mean(empty(:, :, P3_analysis), 3);
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\4.H_Congruent\RBD'
load('avg.mat')
empty = [];
for i = 1:size(all_subject, 2)
    empty(i, :, :) = all_subject{i}.avg;
end
d = data_after_grandavg.avg(Fz, :);
d_P3 = mean(squeeze(empty(:, Fz, P3_analysis)), 2);
d_P3_mean_amplitude1 = mean(empty(:, :, P3_analysis_500_to_600), 3);
d_P3_mean_amplitude2 = mean(empty(:, :, P3_analysis_600_to_700), 3);
d_P3_mean_amplitude3 = mean(empty(:, :, P3_analysis_700_to_800), 3);
dd = data_after_grandavg.avg(Pz, :);
dd_P3 = mean(squeeze(empty(:, Pz, P3_analysis)), 2);
dd_P3_mean_amplitude = mean(empty(:, :, P3_analysis), 3);

[h_RBD_Fz, p_RBD_Fz, ci_RBD_Fz, stats_RBD_Fz] = ttest(c_P3, d_P3, 'Alpha', 0.001);
[h_RBD_Pz, p_RBD_Pz, ci_RBD_Pz, stats_RBD_Pz] = ttest(cc_P3, dd_P3, 'Alpha', 0.001);

% plotting
figure;
subplot(2, 3, 1)
% topoplot(t_value_beta,chanlocs,'emarker2',{find(h_beta),'o','w',10},'maplimits',[0,3]);colorbar;
%topoplot(mean(a_P3_mean_amplitude1, 1),chanlocs,'emarker2',{find(ones(60, 1)),'o','k',5}, 'maplimits',[-0.6,0.6]);
topoplot(mean(a_P3_mean_amplitude1, 1),chanlocs, 'maplimits',[-0.4,0.4]);
subplot(2, 3, 2)
topoplot(mean(a_P3_mean_amplitude2, 1),chanlocs,'maplimits',[-0.4,0.4]);
subplot(2, 3, 3)
topoplot(mean(a_P3_mean_amplitude3, 1),chanlocs,'maplimits',[-0.4,0.4]);colorbar;
subplot(2, 3, 4)
topoplot(mean(b_P3_mean_amplitude1, 1),chanlocs,'maplimits',[-0.4,0.4]);
subplot(2, 3, 5)
topoplot(mean(b_P3_mean_amplitude2, 1),chanlocs,'maplimits',[-0.4,0.4]);
subplot(2, 3, 6)
topoplot(mean(b_P3_mean_amplitude3, 1),chanlocs,'maplimits',[-0.4,0.4]);colorbar;

figure;
subplot(2, 3, 1)
% topoplot(t_value_beta,chanlocs,'emarker2',{find(h_beta),'o','w',10},'maplimits',[0,3]);colorbar;
%topoplot(mean(a_P3_mean_amplitude1, 1),chanlocs,'emarker2',{find(ones(60, 1)),'o','k',5}, 'maplimits',[-0.6,0.6]);
topoplot(mean(c_P3_mean_amplitude1, 1),chanlocs, 'maplimits',[-0.4,0.4]);
subplot(2, 3, 2)
topoplot(mean(c_P3_mean_amplitude2, 1),chanlocs,'maplimits',[-0.4,0.4]);
subplot(2, 3, 3)
topoplot(mean(c_P3_mean_amplitude3, 1),chanlocs,'maplimits',[-0.4,0.4]);colorbar;
subplot(2, 3, 4)
topoplot(mean(d_P3_mean_amplitude1, 1),chanlocs,'maplimits',[-0.4,0.4]);
subplot(2, 3, 5)
topoplot(mean(d_P3_mean_amplitude2, 1),chanlocs,'maplimits',[-0.4,0.4]);
subplot(2, 3, 6)
topoplot(mean(d_P3_mean_amplitude3, 1),chanlocs,'maplimits',[-0.4,0.4]);colorbar;

% figure;
% subplot(2, 2, 1)
% bar([mean(a_P3, 1), mean(b_P3, 1)], erf()
% subplot(2, 2, 2)
% bar([mean(a_P3, 1), mean(b_P3, 1)])


figure;
subplot(2, 1, 1)
plot(tn_target, a(samples_analysis), 'LineWidth',2)
xlim([-800 1000])
ylim([-1.5 1.5])
hold on
plot(tn_target, b(samples_analysis), 'LineWidth',2)
xlim([-800 1000])
ylim([-1.5 1.5])
xline(0, '-', {'target', 'onset'})
xline(-500, ':', {'cue', 'onset'})
yline(0)
legend('Incongruent', 'Congruent')
xlabel('time[ms]')
ylabel('amplitude[uV]')
hold off
subplot(2, 1, 2)
plot(tn_target, aa(samples_analysis), 'LineWidth',2) % 12 30 47
xlim([-800 1000])
ylim([-1.5 1.5])
hold on
plot(tn_target, bb(samples_analysis), 'LineWidth',2)
xlim([-800 1000])
ylim([-1.5 1.5])
xline(0, '-', {'target', 'onset'})
xline(-500, ':', {'cue', 'onset'})
yline(0)
legend('Incongruent', 'Congruent')
xlabel('time[ms]')
ylabel('amplitude[uV]')
hold off

figure;
subplot(2, 1, 1)
plot(tn_target, c(samples_analysis), 'LineWidth',2)
xlim([-800 1000])
ylim([-1.5 1.5])
hold on
plot(tn_target, d(samples_analysis), 'LineWidth',2)
xlim([-800 1000])
ylim([-1.5 1.5])
xline(0, '-', {'target', 'onset'})
xline(-500, ':', {'cue', 'onset'})
yline(0)
legend('Incongruent', 'Congruent')
xlabel('time[ms]')
ylabel('amplitude[uV]')
hold off
subplot(2, 1, 2)
plot(tn_target, cc(samples_analysis), 'LineWidth',2)
xlim([-800 1000])
ylim([-1.5 1.5])
hold on
plot(tn_target, dd(samples_analysis), 'LineWidth',2)
xlim([-800 1000])
ylim([-1.5 1.5])
xline(0, '-', {'target', 'onset'})
xline(-500, ':', {'cue', 'onset'})
yline(0)
legend('Incongruent', 'Congruent')
xlabel('time[ms]')
ylabel('amplitude[uV]')
hold off