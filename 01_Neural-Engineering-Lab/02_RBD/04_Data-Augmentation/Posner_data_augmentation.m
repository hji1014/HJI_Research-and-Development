%% 주의사항

%% Initialize
clc; close all; clear;

%%  (ch * samples * trial)로 변환

con_ch = reshape(con, [280, 60, 29462]);        % reshape
tp_con_ch = permute(con_ch, [2 1 3]);           % 차원 치환
rbd_ch = reshape(rbd, [280, 60, 138400]);
tp_rbd_ch = permute(rbd_ch, [2 1 3]);

% method(d) save matrix
con_m_d_p = zeros(60, 280, 29462);
con_m_d_m = zeros(60, 280, 29462);
rbd_m_d_p = zeros(60, 280, 138400);
rbd_m_d_m = zeros(60, 280, 138400);

% RBD 구간 : 200-250ms, modify 구간 : 25ms-425ms
Fs = 200;
tn = linspace(0, 1400, Fs*1.4);
peak_range = 6:86;                      % 25ms-425ms

%% method(d) : Peak-Amp

fprintf('method(d) 진행중...\n')

% con
for i = 1:60
    for j = 1:29462
        con_m_d_p(i, :, j) = tp_con_ch(i, :, j);
        con_m_d_m(i, :, j) = tp_con_ch(i, :, j);
        c_p = tp_con_ch(i, peak_range, j) * 1.1;          % 110%
        c_m = tp_con_ch(i, peak_range, j) * 0.9;          % 90%
        con_m_d_p(i, peak_range, j) = c_p;
        con_m_d_m(i, peak_range, j) = c_m;
    end
end

for k = 1:60
    for l = 1:138400
        rbd_m_d_p(k, :, l) = tp_rbd_ch(k, :, l);
        rbd_m_d_m(k, :, l) = tp_rbd_ch(k, :, l);
        r_p = tp_rbd_ch(k, peak_range, l) * 1.1;          % 110%
        r_m = tp_rbd_ch(k, peak_range, l) * 0.9;          % 90%
        rbd_m_d_p(k, peak_range, l) = r_p;
        rbd_m_d_m(k, peak_range, l) = r_m;
    end
end

%%

plot(a0)
hold on
plot(b0)