%% Orienting effect

% Initialize
clc; close all; clear;

cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT'
chanlocs = importdata('chanlocs_60.mat');
%parietal_ch = [45, 47, 49, 53, 57, 58, 60];
parietal_ch = [45, 47, 49, 53, 57, 58, 60];
%parietal_ch = [58, 60];         %O1,O2
%parietal_ch = [53, 57];         %PO7,PO8
Fz_Cz_Pz = [12, 30, 47];

Fs = 400;
tn = linspace(-400, 2200, Fs*2.6).';
tn2 = linspace(-900, 1700, Fs*2.6).';
% Center cue
% CON
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\2.H_Center_cue\Control'
con_cc = importdata('ANT_AVG.mat');
con_cc = con_cc.Ensemble_AVG;
con_cc = permute(con_cc, [3 1 2]);
% RBD
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\2.H_Center_cue\RBD'
rbd_cc = importdata('ANT_AVG.mat');
rbd_cc = rbd_cc.Ensemble_AVG;
rbd_cc = permute(rbd_cc, [3 1 2]);

% Spatial cue
% CON
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\3.H_Spatial_cue\Control'
con_sc = importdata('ANT_AVG.mat');
con_sc = con_sc.Ensemble_AVG;
con_sc = permute(con_sc, [3 1 2]);
% RBD
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\3.H_Spatial_cue\RBD'
rbd_sc = importdata('ANT_AVG.mat');
rbd_sc = rbd_sc.Ensemble_AVG;
rbd_sc = permute(rbd_sc, [3 1 2]);

% 대응표본 t-검정, MUA(bh)
% CON_orienting
[h_con_orienting, p_con_orienting, ci_con_orienting, stats_con_orienting] = ttest(con_cc, con_sc);
h_con_orienting = squeeze(h_con_orienting);
p_con_orienting = squeeze(p_con_orienting);
[h_MUA_con_orienting, crit_p_MUA_con_orienting, adj_p_MUA_con_orienting] = fdr_bh(p_con_orienting, 0.05);
%[h_MUA_con_orienting, crit_p_MUA_con_orienting] = fdr_bky(p_con_orienting, 0.05);   %bky -> more powerful
% RBD_orienting
[h_rbd_orienting, p_rbd_orienting, ci_rbd_orienting, stats_rbd_orienting] = ttest(rbd_cc, rbd_sc);
h_rbd_orienting = squeeze(h_rbd_orienting);
p_rbd_orienting = squeeze(p_rbd_orienting);
[h_MUA_rbd_orienting, crit_p_MUA_rbd_orienting, adj_p_MUA_rbd_orienting] = fdr_bh(p_rbd_orienting, 0.001);
%[h_MUA_rbd_orienting, crit_p_MUA_rbd_orienting] = fdr_bky(p_rbd_orienting, 0.05);   %bky -> more powerful

% Grand AVG ERP
con_cc_erp = squeeze(mean(con_cc, 1));
con_cc_erp = mean(con_cc_erp(parietal_ch, :), 1).';
con_sc_erp = squeeze(mean(con_sc, 1));
con_sc_erp = mean(con_sc_erp(parietal_ch, :), 1).';
rbd_cc_erp = squeeze(mean(rbd_cc, 1));
rbd_cc_erp = mean(rbd_cc_erp(parietal_ch, :), 1).';
rbd_sc_erp = squeeze(mean(rbd_sc, 1));
rbd_sc_erp = mean(rbd_sc_erp(parietal_ch, :), 1).';

tn3 = tn(41:680); % -300ms~1300ms

% plotting
% CON
figure(1)           % CON_orienting
subplot(3,1,1);
plot(tn3, con_cc_erp(41:680), 'r', 'LineWidth',2)
hold on
plot(tn3, con_sc_erp(41:680), 'b', 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time(ms)')
ylabel('amplitude')
legend('Center Cue', 'Spatial Cue')
title('정상인 Orienting effect ERP파형 (Grand average : P3,Pz,P4,PO7,PO8,O1,O2)')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
hold off
subplot(3,1,2);
%pcolor(h_con_orienting)
imagesc(h_con_orienting)
colormap('gray');
xticks([41 161 261 361 401 441 679])
xticklabels({'-300ms','0ms','250ms', '500ms', '700ms', '800ms', '1300ms'})
xlabel('time(ms)')
ylabel('channels')
title('paired t-test of center cue-spatial cue')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
subplot(3,1,3);
%pcolor(h_MUA_con_orienting)
imagesc(h_MUA_con_orienting)
colormap('gray');
xticks([41 161 261 361 401 441 679])
xticklabels({'-300ms','0ms','250ms', '500ms', '700ms', '800ms', '1300ms'})
xlabel('time(ms)')
ylabel('channels')
title('after fdr correction by Benjamini&Hochberg')
ax = gca;
ax.TitleFontSizeMultiplier = 2;

% RBD
figure(2)           % CON_orienting
subplot(3,1,1);
plot(tn3, rbd_cc_erp(41:680), 'r', 'LineWidth',2)
hold on
plot(tn3, rbd_sc_erp(41:680), 'b', 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time(ms)')
ylabel('amplitude')
legend('Center Cue', 'Spatial Cue')
title('iRBD - Orienting effect ERP파형 (Grand average : P3,Pz,P4,PO7,PO8,O1,O2)')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
hold off
subplot(3,1,2);
%pcolor(h_rbd_orienting)
imagesc(h_rbd_orienting)
colormap('gray');
xticks([41 161 261 361 401 441 679])
xticklabels({'-300ms','0ms','250ms', '500ms', '700ms', '800ms', '1300ms'})
xlabel('time(ms)')
ylabel('channels')
title('paired t-test of center cue-spatial cue')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
subplot(3,1,3);
%pcolor(h_MUA_rbd_orienting)
imagesc(h_MUA_rbd_orienting)
colormap('gray');
xticks([41 161 261 361 401 441 679])
xticklabels({'-300ms','0ms','250ms', '500ms', '700ms', '800ms', '1300ms'})
xlabel('time(ms)')
ylabel('channels')
title('after fdr correction by Benjamini&Hochberg')
ax = gca;
ax.TitleFontSizeMultiplier = 2;

%% alerting effect

% Initialize
clc; close all; clear;

cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT'
chanlocs = importdata('chanlocs_60.mat');
%parietal_ch = [45, 47, 49, 53, 57, 58, 60];
%parietal_ch = [58, 60];         %O1,O2
parietal_ch = [53, 57];         %PO7,PO8
Fz_Cz_Pz = [12, 30, 47];

Fs = 400;
tn = linspace(-400, 2200, Fs*2.6).';
tn2 = linspace(-900, 1700, Fs*2.6).';
% Center cue
% CON
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\2.H_Center_cue\Control'
con_cc = importdata('ANT_AVG.mat');
con_cc = con_cc.Ensemble_AVG;
con_cc = permute(con_cc, [3 1 2]);
% RBD
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\2.H_Center_cue\RBD'
rbd_cc = importdata('ANT_AVG.mat');
rbd_cc = rbd_cc.Ensemble_AVG;
rbd_cc = permute(rbd_cc, [3 1 2]);

% Spatial cue
% CON
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\1.H_NO_cue\Control'
con_sc = importdata('ANT_AVG.mat');
con_sc = con_sc.Ensemble_AVG;
con_sc = permute(con_sc, [3 1 2]);
% RBD
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\1.H_NO_cue\RBD'
rbd_sc = importdata('ANT_AVG.mat');
rbd_sc = rbd_sc.Ensemble_AVG;
rbd_sc = permute(rbd_sc, [3 1 2]);

% 대응표본 t-검정, MUA(bh)
% CON_orienting
[h_con_orienting, p_con_orienting, ci_con_orienting, stats_con_orienting] = ttest(con_cc, con_sc);
h_con_orienting = squeeze(h_con_orienting);
p_con_orienting = squeeze(p_con_orienting);
[h_MUA_con_orienting, crit_p_MUA_con_orienting, adj_p_MUA_con_orienting] = fdr_bh(p_con_orienting, 0.05);
%[h_MUA_con_orienting, crit_p_MUA_con_orienting] = fdr_bky(p_con_orienting, 0.05);   %bky -> more powerful
% RBD_orienting
[h_rbd_orienting, p_rbd_orienting, ci_rbd_orienting, stats_rbd_orienting] = ttest(rbd_cc, rbd_sc);
h_rbd_orienting = squeeze(h_rbd_orienting);
p_rbd_orienting = squeeze(p_rbd_orienting);
[h_MUA_rbd_orienting, crit_p_MUA_rbd_orienting, adj_p_MUA_rbd_orienting] = fdr_bh(p_rbd_orienting, 0.001);
%[h_MUA_rbd_orienting, crit_p_MUA_rbd_orienting] = fdr_bky(p_rbd_orienting, 0.05);   %bky -> more powerful

% Grand AVG ERP
con_cc_erp = squeeze(mean(con_cc, 1));
con_cc_erp = mean(con_cc_erp(parietal_ch, :), 1).';
con_sc_erp = squeeze(mean(con_sc, 1));
con_sc_erp = mean(con_sc_erp(parietal_ch, :), 1).';
rbd_cc_erp = squeeze(mean(rbd_cc, 1));
rbd_cc_erp = mean(rbd_cc_erp(parietal_ch, :), 1).';
rbd_sc_erp = squeeze(mean(rbd_sc, 1));
rbd_sc_erp = mean(rbd_sc_erp(parietal_ch, :), 1).';

tn3 = tn(41:680); % -300ms~1300ms

% plotting
% CON
figure(1)           % CON_orienting
subplot(3,1,1);
plot(tn3, con_cc_erp(41:680), 'r', 'LineWidth',2)
hold on
plot(tn3, con_sc_erp(41:680), 'k', 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time(ms)')
ylabel('amplitude')
legend('Center Cue', 'No Cue')
title('정상인 Alerting effect ERP파형 (Grand average : P3,Pz,P4,PO7,PO8,O1,O2)')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
hold off
subplot(3,1,2);
%pcolor(h_con_orienting)
imagesc(logical(h_con_orienting))
colormap('gray');
xticks([41 161 261 361 401 441 679])
xticklabels({'-300ms','0ms','250ms', '500ms', '700ms', '800ms', '1300ms'})
xlabel('time(ms)')
ylabel('channels')
title('paired t-test of center cue-no cue')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
subplot(3,1,3);
%pcolor(h_MUA_con_orienting)
imagesc(logical(h_MUA_con_orienting))
colormap('gray');
xticks([41 161 261 361 401 441 679])
xticklabels({'-300ms','0ms','250ms', '500ms', '700ms', '800ms', '1300ms'})
xlabel('time(ms)')
ylabel('channels')
title('after fdr correction by Benjamini&Hochberg')
ax = gca;
ax.TitleFontSizeMultiplier = 2;

% RBD
figure(2)           % CON_orienting
subplot(3,1,1);
plot(tn3, rbd_cc_erp(41:680), 'r', 'LineWidth',2)
hold on
plot(tn3, rbd_sc_erp(41:680), 'k', 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time(ms)')
ylabel('amplitude')
legend('Center Cue', 'No Cue')
title('iRBD - Alerting effect ERP파형 (Grand average : P3,Pz,P4,PO7,PO8,O1,O2)')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
hold off
subplot(3,1,2);
%pcolor(h_rbd_orienting)
imagesc(logical(h_rbd_orienting))
colormap('gray');
xticks([41 161 261 361 401 441 679])
xticklabels({'-300ms','0ms','250ms', '500ms', '700ms', '800ms', '1300ms'})
xlabel('time(ms)')
ylabel('channels')
title('paired t-test of center cue-no cue')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
subplot(3,1,3);
%pcolor(h_MUA_rbd_orienting)
imagesc(logical(h_MUA_rbd_orienting))
colormap('gray');
xticks([41 161 261 361 401 441 679])
xticklabels({'-300ms','0ms','250ms', '500ms', '700ms', '800ms', '1300ms'})
xlabel('time(ms)')
ylabel('channels')
title('after fdr correction by Benjamini&Hochberg')
ax = gca;
ax.TitleFontSizeMultiplier = 2;

%% conflict effect(inhibition)

% Initialize
clc; close all; clear;

cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT'
chanlocs = importdata('chanlocs_60.mat');
%parietal_ch = [45, 47, 49, 53, 57, 58, 60];
%parietal_ch = [58, 60];         %O1,O2
%Fz_Cz_Pz = [12, 30, 47];
%Fz = 12;
%Cz = 30;
Pz = 47;

Fs = 400;
tn = linspace(-400, 2200, Fs*2.6).';
tn2 = linspace(-900, 1700, Fs*2.6).';
% incongruent
% CON
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\5.H_Incongruent\Control'
con_cc = importdata('ANT_AVG.mat');
con_cc = con_cc.Ensemble_AVG;
con_cc = permute(con_cc, [3 1 2]);
% RBD
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\5.H_Incongruent\RBD'
rbd_cc = importdata('ANT_AVG.mat');
rbd_cc = rbd_cc.Ensemble_AVG;
rbd_cc = permute(rbd_cc, [3 1 2]);

% Spatial cue
% CON
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\4.H_Congruent\Control'
con_sc = importdata('ANT_AVG.mat');
con_sc = con_sc.Ensemble_AVG;
con_sc = permute(con_sc, [3 1 2]);
% RBD
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\4.H_Congruent\RBD'
rbd_sc = importdata('ANT_AVG.mat');
rbd_sc = rbd_sc.Ensemble_AVG;
rbd_sc = permute(rbd_sc, [3 1 2]);

% 대응표본 t-검정, MUA(bh)
% CON_orienting
[h_con_orienting, p_con_orienting, ci_con_orienting, stats_con_orienting] = ttest(con_cc, con_sc);
h_con_orienting = squeeze(h_con_orienting);
p_con_orienting = squeeze(p_con_orienting);
[h_MUA_con_orienting, crit_p_MUA_con_orienting, adj_p_MUA_con_orienting] = fdr_bh(p_con_orienting, 0.05);
%[h_MUA_con_orienting, crit_p_MUA_con_orienting] = fdr_bky(p_con_orienting, 0.05);   %bky -> more powerful
% RBD_orienting
[h_rbd_orienting, p_rbd_orienting, ci_rbd_orienting, stats_rbd_orienting] = ttest(rbd_cc, rbd_sc);
h_rbd_orienting = squeeze(h_rbd_orienting);
p_rbd_orienting = squeeze(p_rbd_orienting);
[h_MUA_rbd_orienting, crit_p_MUA_rbd_orienting, adj_p_MUA_rbd_orienting] = fdr_bh(p_rbd_orienting, 0.05);
%[h_MUA_rbd_orienting, crit_p_MUA_rbd_orienting] = fdr_bky(p_rbd_orienting, 0.05);   %bky -> more powerful

% Grand AVG ERP
%con_cc_erp = squeeze(mean(con_cc, 1));
%con_cc_erp = mean(con_cc_erp(Fz_Cz_Pz, :), 1).';
%con_sc_erp = squeeze(mean(con_sc, 1));
%con_sc_erp = mean(con_sc_erp(Fz_Cz_Pz, :), 1).';
%rbd_cc_erp = squeeze(mean(rbd_cc, 1));
%rbd_cc_erp = mean(rbd_cc_erp(Fz_Cz_Pz, :), 1).';
%rbd_sc_erp = squeeze(mean(rbd_sc, 1));
%rbd_sc_erp = mean(rbd_sc_erp(Fz_Cz_Pz, :), 1).';

con_cc_erp = squeeze(mean(con_cc, 1));
con_cc_erp = con_cc_erp(Pz, :).';
con_sc_erp = squeeze(mean(con_sc, 1));
con_sc_erp = con_sc_erp(Pz, :).';
rbd_cc_erp = squeeze(mean(rbd_cc, 1));
rbd_cc_erp = rbd_cc_erp(Pz, :).';
rbd_sc_erp = squeeze(mean(rbd_sc, 1));
rbd_sc_erp = rbd_sc_erp(Pz, :).';

tn3 = tn(41:680); % -300ms~1300ms

% plotting
% CON
figure(1)           % CON_orienting
subplot(3,1,1);
plot(tn3, con_cc_erp(41:680), 'LineWidth',2)
hold on
plot(tn3, con_sc_erp(41:680), 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time(ms)')
ylabel('amplitude')
legend('Incongruent', 'Congruent')
title('정상인 Conflict effect(Inhibition) ERP파형 (Grand average : Pz)')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
hold off
subplot(3,1,2);
%pcolor(h_con_orienting)
imagesc(logical(h_con_orienting))
colormap('gray');
xticks([41 161 261 361 401 481 640])
xticklabels({'-300ms','0ms','250ms', '500ms', '700ms', '900ms', '1300ms'})
xlabel('time(ms)')
ylabel('channels')
title('paired t-test of Incongruent-Congruent')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
subplot(3,1,3);
%pcolor(h_MUA_con_orienting)
imagesc(logical(h_MUA_con_orienting))
colormap('gray');
xticks([41 161 261 361 401 481 640])
xticklabels({'-300ms','0ms','250ms', '500ms', '700ms', '900ms', '1300ms'})
xlabel('time(ms)')
ylabel('channels')
title('after fdr correction by Benjamini&Hochberg')
ax = gca;
ax.TitleFontSizeMultiplier = 2;

% RBD
figure(2)           % CON_orienting
subplot(3,1,1);
plot(tn3, rbd_cc_erp(41:680),'LineWidth',2)
hold on
plot(tn3, rbd_sc_erp(41:680), 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time(ms)')
ylabel('amplitude')
legend('Incongruent', 'Congruent')
title('iRBD - Conflict effect(Inhibition) ERP파형 (Grand average : Pz)')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
hold off
subplot(3,1,2);
%pcolor(h_rbd_orienting)
imagesc(logical(h_rbd_orienting))
colormap('gray');
xticks([41 161 261 361 401 481 640])
xticklabels({'-300ms','0ms','250ms', '500ms', '700ms', '900ms', '1300ms'})
xlabel('time(ms)')
ylabel('channels')
title('paired t-test of Incongruent-Congruent')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
subplot(3,1,3);
%pcolor(h_MUA_rbd_orienting)
imagesc(logical(h_MUA_rbd_orienting))
colormap('gray');
xticks([41 161 261 361 401 481 640])
xticklabels({'-300ms','0ms','250ms', '500ms', '700ms', '900ms', '1300ms'})
xlabel('time(ms)')
ylabel('channels')
title('after fdr correction by Benjamini&Hochberg')
ax = gca;
ax.TitleFontSizeMultiplier = 2;

%% conflict 다른 채널 erp plotting (Fz)

% Initialize
clc; close all; clear;

cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT'
chanlocs = importdata('chanlocs_60.mat');
parietal_ch = [45, 47, 49, 53, 57, 58, 60];
%parietal_ch = [58, 60];         %O1,O2
%Fz_Cz_Pz = [12, 30, 47];
Fz = 12;
CPz_Pz_POz = [39, 47, 55];

Fs = 400;
tn = linspace(-400, 2200, Fs*2.6).';
tn2 = linspace(-900, 1700, Fs*2.6).';
% incongruent
% CON
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\5.H_Incongruent\Control'
con_cc = importdata('ANT_AVG.mat');
con_cc = con_cc.Ensemble_AVG;
con_cc = permute(con_cc, [3 1 2]);
% RBD
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\5.H_Incongruent\RBD'
rbd_cc = importdata('ANT_AVG.mat');
rbd_cc = rbd_cc.Ensemble_AVG;
rbd_cc = permute(rbd_cc, [3 1 2]);

% congruent
% CON
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\4.H_Congruent\Control'
con_sc = importdata('ANT_AVG.mat');
con_sc = con_sc.Ensemble_AVG;
con_sc = permute(con_sc, [3 1 2]);
% RBD
cd 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\4.H_Congruent\RBD'
rbd_sc = importdata('ANT_AVG.mat');
rbd_sc = rbd_sc.Ensemble_AVG;
rbd_sc = permute(rbd_sc, [3 1 2]);

% 대응표본 t-검정, MUA(bh)
% CON_orienting
[h_con_orienting, p_con_orienting, ci_con_orienting, stats_con_orienting] = ttest(con_cc, con_sc);
h_con_orienting = squeeze(h_con_orienting);
p_con_orienting = squeeze(p_con_orienting);
[h_MUA_con_orienting, crit_p_MUA_con_orienting, adj_p_MUA_con_orienting] = fdr_bh(p_con_orienting, 0.05);
%[h_MUA_con_orienting, crit_p_MUA_con_orienting] = fdr_bky(p_con_orienting, 0.05);   %bky -> more powerful
% RBD_orienting
[h_rbd_orienting, p_rbd_orienting, ci_rbd_orienting, stats_rbd_orienting] = ttest(rbd_cc, rbd_sc);
h_rbd_orienting = squeeze(h_rbd_orienting);
p_rbd_orienting = squeeze(p_rbd_orienting);
[h_MUA_rbd_orienting, crit_p_MUA_rbd_orienting, adj_p_MUA_rbd_orienting] = fdr_bh(p_rbd_orienting, 0.001);
%[h_MUA_rbd_orienting, crit_p_MUA_rbd_orienting] = fdr_bky(p_rbd_orienting, 0.05);   %bky -> more powerful

% Grand AVG ERP
%con_cc_erp = squeeze(mean(con_cc, 1));
%con_cc_erp = mean(con_cc_erp(Fz_Cz_Pz, :), 1).';
%con_sc_erp = squeeze(mean(con_sc, 1));
%con_sc_erp = mean(con_sc_erp(Fz_Cz_Pz, :), 1).';
%rbd_cc_erp = squeeze(mean(rbd_cc, 1));
%rbd_cc_erp = mean(rbd_cc_erp(Fz_Cz_Pz, :), 1).';
%rbd_sc_erp = squeeze(mean(rbd_sc, 1));
%rbd_sc_erp = mean(rbd_sc_erp(Fz_Cz_Pz, :), 1).';

con_cc_erp = squeeze(mean(con_cc, 1));
con_cc_erp = con_cc_erp(Fz, :).';
con_sc_erp = squeeze(mean(con_sc, 1));
con_sc_erp = con_sc_erp(Fz, :).';
rbd_cc_erp = squeeze(mean(rbd_cc, 1));
rbd_cc_erp = rbd_cc_erp(Fz, :).';
rbd_sc_erp = squeeze(mean(rbd_sc, 1));
rbd_sc_erp = rbd_sc_erp(Fz, :).';

tn3 = tn(41:680); % -300ms~1300ms

% plotting
% CON
figure(1)           % CON_orienting
subplot(3,1,1);
plot(tn3, con_cc_erp(41:680), 'LineWidth',2)
hold on
plot(tn3, con_sc_erp(41:680), 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time(ms)')
ylabel('amplitude[uV]')
legend('Incongruent', 'Congruent')
title('정상인 Conflict effect(Inhibition) ERP파형 (Grand average : Fz)')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
hold off
subplot(3,1,2);
%pcolor(h_con_orienting)
imagesc(logical(h_con_orienting))
colormap('gray');
xticks([41 161 261 361 401 481 640])
xticklabels({'-300ms','0ms','250ms', '500ms', '700ms', '900ms', '1300ms'})
xlabel('time(ms)')
ylabel('channels')
title('paired t-test of Incongruent-Congruent')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
subplot(3,1,3);
%pcolor(h_MUA_con_orienting)
imagesc(logical(h_MUA_con_orienting))
colormap('gray');
xticks([41 161 261 361 401 481 640])
xticklabels({'-300ms','0ms','250ms', '500ms', '700ms', '900ms', '1300ms'})
xlabel('time(ms)')
ylabel('channels')
title('after fdr correction by Benjamini&Hochberg')
ax = gca;
ax.TitleFontSizeMultiplier = 2;

% RBD
figure(2)           % CON_orienting
subplot(3,1,1);
plot(tn3, rbd_cc_erp(41:680),'LineWidth',2)
hold on
plot(tn3, rbd_sc_erp(41:680), 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time(ms)')
ylabel('amplitude[uV]')
legend('Incongruent', 'Congruent')
title('iRBD - Conflict effect(Inhibition) ERP파형 (Grand average : Fz)')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
hold off
subplot(3,1,2);
%pcolor(h_rbd_orienting)
imagesc(logical(h_rbd_orienting))
colormap('gray');
xticks([41 161 261 361 401 481 640])
xticklabels({'-300ms','0ms','250ms', '500ms', '700ms', '900ms', '1300ms'})
xlabel('time(ms)')
ylabel('channels')
title('paired t-test of Incongruent-Congruent')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
subplot(3,1,3);
%pcolor(h_MUA_rbd_orienting)
imagesc(logical(h_MUA_rbd_orienting))
colormap('gray');
xticks([41 161 261 361 401 481 640])
xticklabels({'-300ms','0ms','250ms', '500ms', '700ms', '900ms', '1300ms'})
xlabel('time(ms)')
ylabel('channels')
title('after fdr correction by Benjamini&Hochberg')
ax = gca;
ax.TitleFontSizeMultiplier = 2;