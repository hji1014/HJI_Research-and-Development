%%
clear; close all; clc;

%% permutation test example
% rng(0)
% 
% X = rand(6, 1);
% t_perm = zeros(5000, 1);
% 
% for i_perm = 1:5000
%     X_shuffle = X(randperm(6));
%     
%     X1 = X_shuffle(1:3);
%     X2 = X_shuffle(4:6);
%     
%     [~, ~, ~, STATS] = ttest(X1, X2);
%     
%     t_perm(i_perm) = STATS.tstat;
%     
% end
% 
% hist(t_perm);

%% MUA
Fs = 400;
tn = linspace(-400, 2200, Fs*2.6).';
chanlocs = importdata('C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\chanlocs_60.mat');

data_1 = importdata('D:/ANT_ERP/con_nc_ensemble_avg.mat');
data_2 = importdata('D:/ANT_ERP/con_cc_ensemble_avg.mat');

% 여동훈 선배 matrix_wise t-test
% for i = 1:4
%     importdata('D:\ANT_ERP_MUA_cluster_based_permutation_test/con_alerting_cluster_results.mat'){i};
% end
% s = [];
% s(1, :, :, :) = data_1;
% s(2, :, :, :) = data_2;
% s = permute(s, [4 1 2 3]);
% s1 = squeeze(s(:, 1, :, :));
% s2 = squeeze(s(:, 2, :, :));
% [t_observed,P_observed]=ttest_mtx_wise_ERP(s);      %  여동훈 선배 코드

[h, p, ci, stats] = ttest(permute(data_1,[3 1 2]), permute(data_2,[3 1 2]));    % 위와 같은 결과 (uncorrected paired t-test)
t_val = squeeze(stats.tstat);
p_val = squeeze(p);
h_val = squeeze(h);

%%% Bonferroni correction
h_bonferroni = double(p_val < (0.05/(60*1040)));

%%% "two-stage" Benjamini, Krieger, & Yekutieli (2006) (fdr_bky)
[h_bky, crit_p_bky]=fdr_bky(p_val);

%%% Benjamini & Hochberg (1995)
[h_bh, crit_p_bh, adj_p_bh]=fdr_bh(p_val);

%%% Benjamini&Yekutieli (2001)
%[h_by, crit_p_by, adj_p_by]=fdr_bh(p_val, method, 'dep');

%% 위키독스 참고한 cluster-based permutation test (ERP)
Fs = 400;
tn = linspace(-400, 2200, Fs*2.6).';
chanlocs = importdata('C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\chanlocs_60.mat');

data_1 = importdata('D:/ANT_ERP/con_ic_ensemble_avg.mat');
data_2 = importdata('D:/ANT_ERP/con_cg_ensemble_avg.mat');

[h, p, ci, stats] = ttest(permute(data_1,[3 1 2]), permute(data_2,[3 1 2]));    % uncorrected paired t-test
t_val = squeeze(stats.tstat);
p_val = squeeze(p);
h_val = squeeze(h);

ROI_hood = find_chan_hood_ERP(chanlocs);            % 60ch 중 38mm 거리 이하인 전극을 인접한 전극이라고 표시(Kutas et al), symmetric matrix
%채널 위치 확인 topoplot
%a = zeros(60, 1);
%topoplot(a, chanlocs, 'emarker2',{find(a==0),'o','w',15})
df = size(data_1, 3)-1;                                % 자유도
P_thresh = 0.05;
T_thresh = tinv(1-P_thresh/2, df);      % 양측 5% 유의수준에서 t값 계산

% 클러스터 구하기
[clust_membership, n_clust] = find_clusters_ERP_t(abs(t_val), T_thresh, ROI_hood);

% cluster membership of uncorreted t-value
figure(1)
imagesc(clust_membership)
xticks([1 41 81 121 161 201 241 281 321 361 401 441 481 521 561 601 641 681 721 761 801 841 881 921 961 1001 1040])
xticklabels({'-400','-300','-200','-100','0','100','200','300','400','500','600','700','800','900','1000', ...
   '1100','1200','1300','1400','1500','1600','1700','1800','1900','2000','2100','2200'})
colormap('jet');
colorbar;

% 각 클러스터의 mass 계산
mass_clust = [];
for i_clust = 1 : n_clust
    idx = find(clust_membership==i_clust);
    mass_clust(i_clust) = sum(abs(t_val(idx)));
end
max_mass = max(mass_clust);
max_mass_idx = find(mass_clust == max_mass);

% permutation test
iteration_num = 5000;                            % iteration만큼 permutation test 진행
sub_idx = 1:size(data_1, 3);                    % 피험자 수
exchange_num_max = round(size(data_1, 3)/2);     % 최대 교환 피험자 수 = 전체 피험자의 50%

perm_data_1 = zeros(size(data_1));
perm_data_2 = zeros(size(data_2));

perm_max_cluster = [];

% iteration_num 만큼 permutation test 수행
tic
for i = 1:iteration_num
    perm_data_1 = data_1;
    perm_data_2 = data_2;
    
    exchange_num = randperm(exchange_num_max, 1);
    rand_sub = randperm(size(data_1, 3), exchange_num);                                       % 피험자의 조건을 바꾸기
    rand_sub = sort(rand_sub);
    
    perm_data_1(:, :, rand_sub) = data_2(:, :, rand_sub);
    perm_data_2(:, :, rand_sub) = data_1(:, :, rand_sub);
    
    [h, p, ci, stats] = ttest(permute(perm_data_1,[3 1 2]), permute(perm_data_2,[3 1 2]));    % 위와 같은 결과 (uncorrected paired t-test)
    t_val = squeeze(stats.tstat);
    p_val = squeeze(p);
    h_val = squeeze(h);
    
    [perm_clust_membership, perm_n_clust] = find_clusters_ERP_t(abs(t_val), T_thresh, ROI_hood);        % 클러스터 구하기
    
    perm_mass_clust = [];
    for i_clust = 1 : perm_n_clust
        idx = find(perm_clust_membership==i_clust);
        perm_mass_clust(i_clust) = sum(abs(t_val(idx)));
    end
    perm_max_mass = max(perm_mass_clust);
    perm_max_mass_idx = find(perm_mass_clust == perm_max_mass);
    perm_max_cluster(i) = perm_max_mass;
end
toc

% permutation 분포 기준 유의한 클러스터 찾기
sig_mass_value = quantile(perm_max_cluster, 0.95);
sig_mass_uncorrected_cluster_idx = find(mass_clust >= sig_mass_value);

% p-value 계산
for i=1:length(sig_mass_uncorrected_cluster_idx)
    p_val_sig_clust(i)= (sum(perm_max_cluster>mass_clust(sig_mass_uncorrected_cluster_idx(i)))+1)/iteration_num;
end

save('D:\ANT_ERP_MUA_cluster_based_permutation_test\연구실코드 결과\new_MUA\con_conflict_erp_MUA.mat')

%% MUA 결과 plot
clear; close all; clc;

load('D:\ANT_ERP_MUA_cluster_based_permutation_test\연구실코드 결과\new_MUA\rbd_conflict_erp_MUA.mat');

% 유의한 클러스터의 t-value 표시한 figure
%a = find(clust_membership == 35 | clust_membership == 38);
clear a b
a = find(clust_membership == 75);
b = zeros(60, 1040);
b(a) = t_val(a);

figure(1)
imagesc(b)
xticks([1 41 81 121 161 201 241 281 321 361 401 441 481 521 561 601 641 681 721 761 801 841 881 921 961 1001 1040])
xticklabels({'-400','-300','-200','-100','0','100','200','300','400','500','600','700','800','900','1000', ...
   '1100','1200','1300','1400','1500','1600','1700','1800','1900','2000','2100','2200'})
colormap('jet');
colorbar;
caxis([-6 6])


%% cluster-based permutation test - 오픈소스

% [F_observed,P_observed] = anova1_ERP(s);
% threshold = finv(0.95, 1, 61);
% chanlocs = importdata('C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\chanlocs_60.mat');
% chan_hood=find_chan_hood_ERP(chanlocs);
% 
% [clust_membership, n_clust] = find_clusters_ERP(F_observed, threshold, chan_hood);

% 오픈소스
tic
[clusters, p_values, t_sums, permutation_distribution ] = permutest(data_1, data_2, true, 0.05, 5000);
toc

cluster_plot = zeros(60, 1040);
for i = 1:2
    cluster_plot(clusters{1,i}) = i;
end

figure(2) % cluster membership of uncorreted t-value
imagesc(cluster_plot)
xticks([1 41 81 121 161 201 241 281 321 361 401 441 481 521 561 601 641 681 721 761 801 841 881 921 961 1001 1040])
xticklabels({'-400','-300','-200','-100','0','100','200','300','400','500','600','700','800','900','1000', ...
    '1100','1200','1300','1400','1500','1600','1700','1800','1900','2000','2100','2200'})
colormap('jet');
colorbar;

figure(3) % cluster membership of uncorreted t-value
plot(tn, mean(mean(data_1(40:60, :, :), 3), 1), 'k', 'LineWidth',2)
hold on
plot(tn, mean(mean(data_2(40:60, :, :), 3), 1), 'r', 'LineWidth',2)
legend('No cue', 'Center cue')
hold off

%save('con_alerting_cluster_results.mat', 'clusters', 'p_values', 't_sums', 'permutation_distribution')

%% ERP plot
clear; close all; clc;

Fs = 400;
tn = linspace(-400, 2200, Fs*2.6).';
chanlocs = importdata('C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\chanlocs_60.mat');

% alerting
data_1 = importdata('D:/ANT_ERP/con_nc_ensemble_avg.mat');
data_1 = mean(data_1, 3);
data_2 = importdata('D:/ANT_ERP/con_cc_ensemble_avg.mat');
data_2 = mean(data_2, 3);
data_3 = importdata('D:/ANT_ERP/rbd_nc_ensemble_avg.mat');
data_3 = mean(data_3, 3);
data_4 = importdata('D:/ANT_ERP/rbd_cc_ensemble_avg.mat');
data_4 = mean(data_4, 3);

% orienting
data_1 = importdata('D:/ANT_ERP/con_cc_ensemble_avg.mat');
data_1 = mean(data_1, 3);
data_2 = importdata('D:/ANT_ERP/con_sc_ensemble_avg.mat');
data_2 = mean(data_2, 3);
data_3 = importdata('D:/ANT_ERP/rbd_cc_ensemble_avg.mat');
data_3 = mean(data_3, 3);
data_4 = importdata('D:/ANT_ERP/rbd_sc_ensemble_avg.mat');
data_4 = mean(data_4, 3);

% conflict
data_1 = importdata('D:/ANT_ERP/con_ic_ensemble_avg.mat');
data_1 = mean(data_1, 3);
data_2 = importdata('D:/ANT_ERP/con_cg_ensemble_avg.mat');
data_2 = mean(data_2, 3);
data_3 = importdata('D:/ANT_ERP/rbd_ic_ensemble_avg.mat');
data_3 = mean(data_3, 3);
data_4 = importdata('D:/ANT_ERP/rbd_cg_ensemble_avg.mat');
data_4 = mean(data_4, 3);

% 특정 채널을 topo에서 확인
% a = zeros(1, 60);
% a(10:14) = 1;
% topoplot(a, chanlocs, 'emarker2',{find(a==1),'o','w',15})

%% alerting ERP plot(F, C, P, O)
figure(1)
subplot(4, 2, 1)
plot(tn, mean(data_1(8:16, :), 1), 'k', 'LineWidth',2)
hold on
plot(tn, mean(data_2(8:16, :), 1), 'r', 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('No Cue', 'Center Cue')
title('CON')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off
subplot(4, 2, 2)
plot(tn, mean(data_3(8:16, :), 1), 'k', 'LineWidth',2)
hold on
plot(tn, mean(data_4(8:16, :), 1), 'r', 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('No Cue', 'Center Cue')
title('iRBD')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off

subplot(4, 2, 3)
plot(tn, mean(data_1(27:33, :), 1), 'k', 'LineWidth',2)
hold on
plot(tn, mean(data_2(27:33, :), 1), 'r', 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('No Cue', 'Center Cue')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off
subplot(4, 2, 4)
plot(tn, mean(data_3(27:33, :), 1), 'k', 'LineWidth',2)
hold on
plot(tn, mean(data_4(27:33, :), 1), 'r', 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('No Cue', 'Center Cue')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off

subplot(4, 2, 5)
plot(tn, mean(data_1(43:52, :), 1), 'k', 'LineWidth',2)
hold on
plot(tn, mean(data_2(43:52, :), 1), 'r', 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('No Cue', 'Center Cue')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off
subplot(4, 2, 6)
plot(tn, mean(data_3(43:52, :), 1), 'k', 'LineWidth',2)
hold on
plot(tn, mean(data_4(43:52, :), 1), 'r', 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('No Cue', 'Center Cue')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off

subplot(4, 2, 7)
plot(tn, mean(data_1(58:60, :), 1), 'k', 'LineWidth',2)
hold on
plot(tn, mean(data_2(58:60, :), 1), 'r', 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('No Cue', 'Center Cue')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off
subplot(4, 2, 8)
plot(tn, mean(data_3(58:60, :), 1), 'k', 'LineWidth',2)
hold on
plot(tn, mean(data_4(58:60, :), 1), 'r', 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('No Cue', 'Center Cue')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off

%% orienting ERP plot(F, C, P, O)
figure(1)
subplot(4, 2, 1)
plot(tn, mean(data_1(8:16, :), 1), 'r', 'LineWidth',2)
hold on
plot(tn, mean(data_2(8:16, :), 1), 'b', 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('Center Cue', 'Spatial Cue')
title('CON')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off
subplot(4, 2, 2)
plot(tn, mean(data_3(8:16, :), 1), 'r', 'LineWidth',2)
hold on
plot(tn, mean(data_4(8:16, :), 1), 'b', 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('Center Cue', 'Spatial Cue')
title('iRBD')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off

subplot(4, 2, 3)
plot(tn, mean(data_1(27:33, :), 1), 'r', 'LineWidth',2)
hold on
plot(tn, mean(data_2(27:33, :), 1), 'b', 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('Center Cue', 'Spatial Cue')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off
subplot(4, 2, 4)
plot(tn, mean(data_3(27:33, :), 1), 'r', 'LineWidth',2)
hold on
plot(tn, mean(data_4(27:33, :), 1), 'b', 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('Center Cue', 'Spatial Cue')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off

subplot(4, 2, 5)
plot(tn, mean(data_1(43:52, :), 1), 'r', 'LineWidth',2)
hold on
plot(tn, mean(data_2(43:52, :), 1), 'b', 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('Center Cue', 'Spatial Cue')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off
subplot(4, 2, 6)
plot(tn, mean(data_3(43:52, :), 1), 'r', 'LineWidth',2)
hold on
plot(tn, mean(data_4(43:52, :), 1), 'b', 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('Center Cue', 'Spatial Cue')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off

subplot(4, 2, 7)
plot(tn, mean(data_1(58:60, :), 1), 'r', 'LineWidth',2)
hold on
plot(tn, mean(data_2(58:60, :), 1), 'b', 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('Center Cue', 'Spatial Cue')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off
subplot(4, 2, 8)
plot(tn, mean(data_3(58:60, :), 1), 'r', 'LineWidth',2)
hold on
plot(tn, mean(data_4(58:60, :), 1), 'b', 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('Center Cue', 'Spatial Cue')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off

%% conflict ERP plot(F, C, P, O)
figure(1)
subplot(4, 2, 1)
plot(tn, mean(data_1(8:16, :), 1), 'LineWidth',2)
hold on
plot(tn, mean(data_2(8:16, :), 1), 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('Incongruent target', 'Congruent target')
title('CON')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off
subplot(4, 2, 2)
plot(tn, mean(data_3(8:16, :), 1), 'LineWidth',2)
hold on
plot(tn, mean(data_4(8:16, :), 1), 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('Incongruent target', 'Congruent target')
title('iRBD')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off

subplot(4, 2, 3)
plot(tn, mean(data_1(27:33, :), 1), 'LineWidth',2)
hold on
plot(tn, mean(data_2(27:33, :), 1), 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('Incongruent target', 'Congruent target')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off
subplot(4, 2, 4)
plot(tn, mean(data_3(27:33, :), 1), 'LineWidth',2)
hold on
plot(tn, mean(data_4(27:33, :), 1), 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('Incongruent target', 'Congruent target')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off

subplot(4, 2, 5)
plot(tn, mean(data_1(43:52, :), 1), 'LineWidth',2)
hold on
plot(tn, mean(data_2(43:52, :), 1), 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('Incongruent target', 'Congruent target')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off
subplot(4, 2, 6)
plot(tn, mean(data_3(43:52, :), 1), 'LineWidth',2)
hold on
plot(tn, mean(data_4(43:52, :), 1), 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('Incongruent target', 'Congruent target')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off

subplot(4, 2, 7)
plot(tn, mean(data_1(58:60, :), 1), 'LineWidth',2)
hold on
plot(tn, mean(data_2(58:60, :), 1), 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('Incongruent target', 'Congruent target')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off
subplot(4, 2, 8)
plot(tn, mean(data_3(58:60, :), 1), 'LineWidth',2)
hold on
plot(tn, mean(data_4(58:60, :), 1), 'LineWidth',2)
xline(0,'-',{'cue start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xline(500,'-',{'target start'},'LineWidth',1.5, 'LabelOrientation', 'horizontal');
xlabel('time [ms]')
ylabel('amplitude [uV]')
legend('Incongruent target', 'Congruent target')
ax = gca;
ax.TitleFontSizeMultiplier = 2;
ylim([-2 2])
hold off
%% topography plot

