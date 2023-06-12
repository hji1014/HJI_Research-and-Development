%% Initialize
clc; close all; clear;

%% Data import

% SOA 200ms
valid_200 = load('200_valid.mat').Data;
invalid_200 = load('200_invalid.mat').Data;

% SOA 1000ms
valid_1000 = load('1000_valid.mat').Data;
invalid_1000 = load('1000_invalid.mat').Data;

%% LPF 20Hz & Baseline Correction

Fs = 400;
N = 400;

tn_200 = linspace(-600, 1400, Fs*2);
tn_1000 = linspace(-1600, 1400, Fs*3);

Nf = 7;         % 7차 필터
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

cue_onset_200 = 160;
target_onest_200 = 240;

cue_onset_1000 = 240;
target_onset_1000 = 640;

% valid_200

for i = 1:60        % each ch
  valid_200_ch = valid_200(i, :, :);
  for j = 1:size(valid_200_ch, 3)       % each trial
      valid_200_trial = squeeze(valid_200_ch(:, :, j));
      
      % filtering
      xn = valid_200_trial;
      xfiltfilt = filtfilt(d, xn);
      
      % baseline correction
      xfilt_BC = xfiltfilt - mean(xfiltfilt(1:240));
      
      % integration
      valid_200(i, :, j) = xfilt_BC;
      
  end
end

% invalid_200
for m = 1:60        % each ch
  invalid_200_ch = invalid_200(m, :, :);
  for n = 1:size(invalid_200_ch, 3)       % each trial
      invalid_200_trial = squeeze(invalid_200_ch(:, :, n));
      
      % filtering
      xn = invalid_200_trial;
      xfiltfilt = filtfilt(d, xn);
      
      % baseline correction
      xfilt_BC = xfiltfilt - mean(xfiltfilt(1:240));
      
      % integration
      invalid_200(m, :, n) = xfilt_BC;
      
  end
end
% valid_1000

for i = 1:60        % each ch
  valid_1000_ch = valid_1000(i, :, :);
  for j = 1:size(valid_1000_ch, 3)       % each trial
      valid_1000_trial = squeeze(valid_1000_ch(:, :, j));
      
      % filtering
      xn = valid_1000_trial;
      xfiltfilt = filtfilt(d, xn);
      
      % baseline correction
      xfilt_BC = xfiltfilt - mean(xfiltfilt(1:cue_onset_1000));
      
      % integration
      valid_1000(i, :, j) = xfilt_BC;
      
  end
end

% invalid_1000

for i = 1:60        % each ch
  invalid_1000_ch = invalid_1000(i, :, :);
  for j = 1:size(invalid_1000_ch, 3)       % each trial
      invalid_1000_trial = squeeze(invalid_1000_ch(:, :, j));
      
      % filtering
      xn = invalid_1000_trial;
      xfiltfilt = filtfilt(d, xn);
      
      % baseline correction
      xfilt_BC = xfiltfilt - mean(xfiltfilt(1:cue_onset_1000));
      
      % integration
      invalid_1000(i, :, j) = xfilt_BC;
      
  end
end


%% Average

valid_200_chavg = zeros(60, 800);
invalid_200_chavg = zeros(60, 800);
valid_1000_chavg = zeros(60, 1200);
invalid_1000_chavg = zeros(60, 1200);

% valid_200
for a = 1:60
    valid_200_avg = mean(squeeze(valid_200(a, :, :)), 2);
    valid_200_chavg(a, :) = valid_200_avg;
end

% invalid_200
for b = 1:60
    invalid_200_avg = mean(squeeze(invalid_200(b, :, :)), 2);
    invalid_200_chavg(b, :) = invalid_200_avg;
end

% valid_1000
for a = 1:60
    valid_1000_avg = mean(squeeze(valid_1000(a, :, :)), 2);
    valid_1000_chavg(a, :) = valid_1000_avg;
end

% invalid_1000
for b = 1:60
    invalid_1000_avg = mean(squeeze(invalid_1000(b, :, :)), 2);
    invalid_1000_chavg(b, :) = invalid_1000_avg;
end

%% Plotting
%figure(1)
%plot(tn, xn, tn, xfiltfilt)
%hold on
%plot(tn, xfilt_BC, 'r', 'linewidth', 2)
%hold off
%xline(0,'--k');

% 플롯 여러개 그리기
figure(2)
for plotid = 1 : 15
    subplot(3, 5, plotid);
    plot(tn_200, valid_200_chavg(plotid, :), tn_200, invalid_200_chavg(plotid, :))
    legend('valid 200', 'invalid 200')
    xlim([-600 1400])
    xticks([-600 0 1400])
    xline(0, '--k');
end

figure(3)
for plotid = 1 : 15
    subplot(3, 5, plotid);
    plot(tn_1000, valid_1000_chavg(plotid, :), tn_1000, invalid_1000_chavg(plotid, :))
    legend('valid 1000', 'invalid 1000')
    xlim([-1600 1400])
    xticks([-1600 0 1400])
    xline(0, '--k');
end