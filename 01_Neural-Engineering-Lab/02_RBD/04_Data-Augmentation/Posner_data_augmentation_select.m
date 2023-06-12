%a = rbd_m_d_p;
%b = permute(a, [2 1 3]);
%c = reshape(b, [16800, 138400]);

%% 주의사항

%% Initialize
clc; close all; clear;

%% 랜덤 추출 (k trial)

CON = R_G_all(1:29462, 2);
RBD = R_G_all(29463:end, 2);

% subject 설정
CON_1_trial = find(CON==20);
CON_2_trial = find(CON==40);
RBD_1_trial = find(RBD==80);
RBD_2_trial = find(RBD==160);

% 결과 재현을 위한 난수 시드값
s = RandStream('mlfg6331_64');
k = 40;
CON_1_trial_k = sort(datasample(CON_1_trial, k));
CON_2_trial_k = sort(datasample(CON_2_trial, k));
RBD_1_trial_k = sort(datasample(RBD_1_trial, k));
RBD_2_trial_k = sort(datasample(RBD_2_trial, k));

%% non-augmented data
a = zeros(1, 80);
b = ones(1, 80);
label = horzcat(a, b);

CON_1_non = con(:, CON_1_trial_k);
CON_2_non = con(:, CON_2_trial_k);
RBD_1_non = rbd(:, RBD_1_trial_k);
RBD_2_non = rbd(:, RBD_2_trial_k);

DNN_non = horzcat(CON_1_non, CON_2_non, RBD_1_non, RBD_2_non);                         % 16800 x 160

DNN_non_label = vertcat(DNN_non, label);            % 16801 x 160 (1은 label)
DNN_non_label_final = DNN_non_label.';                                                % 160 x 16801

%% augmented data 합치기

CON_m = permute(con_m_d_m, [2 1 3]);
CON_m_r = reshape(CON_m, [16800, 29462]);
CON_p = permute(con_m_d_p, [2 1 3]);
CON_p_r = reshape(CON_p, [16800, 29462]);
RBD_m = permute(rbd_m_d_m, [2 1 3]);
RBD_m_r = reshape(RBD_m, [16800, 138400]);
RBD_P = permute(rbd_m_d_p, [2 1 3]);
RBD_P_r = reshape(RBD_P, [16800, 138400]);

CON_1_40_m = CON_m_r(:, CON_1_trial_k);
CON_2_40_m = CON_m_r(:, CON_2_trial_k);
CON_1_40_p = CON_p_r(:, CON_1_trial_k);
CON_2_40_p = CON_p_r(:, CON_2_trial_k);
RBD_1_40_m = RBD_m_r(:, RBD_1_trial_k);
RBD_2_40_m = RBD_m_r(:, RBD_2_trial_k);
RBD_1_40_p = RBD_P_r(:, RBD_1_trial_k);
RBD_2_40_p = RBD_P_r(:, RBD_2_trial_k);

DNN_aug = horzcat(DNN_non, CON_1_40_m, CON_2_40_m, CON_1_40_p, CON_2_40_p, RBD_1_40_m, RBD_2_40_m, RBD_1_40_p, RBD_2_40_p);

label_z = zeros(1, 480);
label_z(1, 81:160) = 1;
label_z(1, 321:480) = 1;

DNN_aug_label = vertcat(DNN_aug, label_z);                                               % 16801 x 480
DNN_aug_label_final = DNN_aug_label.';                                                  % 480 x 16801