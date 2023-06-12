%% initialization
clear; close all; clc;

%% edf region 별 평균
eeglab

region = {'PCG';'SP';'PRC';'MCG';'SM';'TP';'AH';'PH';'PHG';'ASI';'MSI';'PSI';'PI'};        % 직접 복붙하기.

% PCG
PCG = mean(EEG.data(3:18, :), 1);
EEG.data = cat(1, EEG.data, PCG);
EEG.nbchan = EEG.nbchan + 1;

% SP
SP = mean(EEG.data(19:34, :), 1);
EEG.data = cat(1, EEG.data, SP);
EEG.nbchan = EEG.nbchan + 1;


% PRC
PRC = mean(EEG.data(35:46, :), 1);
EEG.data = cat(1, EEG.data, PRC);
EEG.nbchan = EEG.nbchan + 1;

% MCG
MCG = mean(EEG.data(48:63, :), 1);
EEG.data = cat(1, EEG.data, MCG);
EEG.nbchan = EEG.nbchan + 1;

% SM
SM = mean(EEG.data(64:77, :), 1);
EEG.data = cat(1, EEG.data, SM);
EEG.nbchan = EEG.nbchan + 1;

% TP
TP = mean(EEG.data(78:85, :), 1);
EEG.data = cat(1, EEG.data, TP);
EEG.nbchan = EEG.nbchan + 1;

% AH
AH = mean(EEG.data(86:97, :), 1);
EEG.data = cat(1, EEG.data, AH);
EEG.nbchan = EEG.nbchan + 1;

% PH
PH = mean(EEG.data(98:109, :), 1);
EEG.data = cat(1, EEG.data, PH);
EEG.nbchan = EEG.nbchan + 1;

% PHG
PHG = mean(EEG.data(110:123, :), 1);
EEG.data = cat(1, EEG.data, PHG);
EEG.nbchan = EEG.nbchan + 1;

% ASI
ASI = mean(EEG.data(124:139, :), 1);
EEG.data = cat(1, EEG.data, ASI);
EEG.nbchan = EEG.nbchan + 1;

% MSI
MSI = mean(EEG.data(140:147, :), 1);
EEG.data = cat(1, EEG.data, MSI);
EEG.nbchan = EEG.nbchan + 1;

% PSI
PSI = mean(EEG.data(148:155, :), 1);
EEG.data = cat(1, EEG.data, PSI);
EEG.nbchan = EEG.nbchan + 1;

% PI
PI = mean(EEG.data(156:163, :), 1);
EEG.data = cat(1, EEG.data, PI);
EEG.nbchan = EEG.nbchan + 1;

%%

