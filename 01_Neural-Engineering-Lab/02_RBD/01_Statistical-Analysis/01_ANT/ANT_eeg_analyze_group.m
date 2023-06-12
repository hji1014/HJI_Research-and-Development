clearvars -except do* Stim_i
outputdir = 'Analysis_EEG\erp'; % Make directory (\images, \grands)
outputdir2 = 'Analysis_EEG\freq'; % Make directory (\images, \grands)
outputdir3 = 'Analysis_EEG\Result_emotion_source_plv'; % Make directory (\images, \grands)

outputdir5 = 'Analysis_EEG\svm'; % Make directory (\images, \grands)
warning on

do_timelock = false;
do_cluster = false;
do_microstate = false;
do_source = false;
do_connectivity = false;
do_connectivity_pcc = false;
do_convergence_individual = false;
do_svm = false;

group_info = [ones(1,17) ones(1,47-17)*2]; % 1: CON, 2: RBD

if do_timelock
    %% ERP analysis
    %% load the single subject averages
    subj = 47;
    Stat_sub = ones(1,subj);
    for i = 1:subj
        details = sprintf('subject%02d', i);
        eval(details);
        load([outputdir filesep subjectdata.subjectnr '_' 'timelock.mat']);
        GA_ERP_NC{i} = ERP_NoCue;
        GA_ERP_CC{i} = ERP_CenterCue;
        GA_ERP_SC{i} = ERP_SpatialCue;
        GA_ERP_Cg{i} = ERP_Cong;
        GA_ERP_ICg{i} = ERP_Incong;
        GA_ERP_alerting{i} = ERP_alerting;
        GA_ERP_orienting{i} = ERP_orienting;
        GA_ERP_conflict{i} = ERP_conflict;
    end
    
    Sub2analysis = 'All';
    switch Sub2analysis
        case 'All'
            Sub_idx = 1:subj;
        case 'CON'
            Sub_idx = find(group_info==1);
        case 'RBD'
            Sub_idx = find(group_info==2);
    end
    % calculate grand average for each condition
    cfg = [];
    cfg.channel   = 'all';
    cfg.latency   = 'all';
    cfg.parameter = 'avg';
    GA_ERP_NC_avg    = ft_timelockgrandaverage(cfg,GA_ERP_NC{Sub_idx});
    GA_ERP_CC_avg  = ft_timelockgrandaverage(cfg,GA_ERP_CC{Sub_idx});
    GA_ERP_SC_avg  = ft_timelockgrandaverage(cfg,GA_ERP_SC{Sub_idx});
    GA_ERP_Cg_avg    = ft_timelockgrandaverage(cfg,GA_ERP_Cg{Sub_idx});
    GA_ERP_ICg_avg  = ft_timelockgrandaverage(cfg,GA_ERP_ICg{Sub_idx});
    
    
    %% 0. visualize pattern
    
    % multiplot
    cfg = [];
    %     cfg.fontsize = 6;
    cfg.layout = 'snuh-60ch';
    %         cfg.xlim = [-0.2 1.5];
    cfg.ylim = [-6 4];
    cfg.showlabels = 'on';
    cfg.interactive = 'yes';
    cfg.showoutline = 'yes';
    
    figure
    ft_multiplotER(cfg, GA_ERP_NC_avg, GA_ERP_CC_avg, GA_ERP_SC_avg );
    legend({'No Cue';'Center Cue';'Spatial Cue'});
    set(gcf,'Position',[1 1 1239 945]);
    
    % set time axis to target-locked style
    GA_ERP_NC_avg.time = GA_ERP_Cg_avg.time;
    GA_ERP_CC_avg.time = GA_ERP_Cg_avg.time;
    GA_ERP_SC_avg.time = GA_ERP_Cg_avg.time;
    % singleplot
    cfg = [];
    cfg.layout = 'snuh-60ch';
    %         cfg.xlim = [-0.2 1.5];
    cfg.ylim = [-3 3];
    cfg.showoutline = 'yes';
    figure
    cfg.channel = 'FZ';
    subplot(1,3,1);
    ft_singleplotER(cfg, GA_ERP_NC_avg, GA_ERP_CC_avg, GA_ERP_SC_avg );
    title(cfg.channel);xlabel('Time (s)');ylabel('Amplitude [uV]');
    set(gca,'fontsize',18);
    cfg.channel = 'CZ';
    subplot(1,3,2);
    ft_singleplotER(cfg, GA_ERP_NC_avg, GA_ERP_CC_avg, GA_ERP_SC_avg );
    title(cfg.channel);xlabel('Time (s)');ylabel('Amplitude [uV]');
    set(gca,'fontsize',18);
    cfg.channel = 'PZ';
    subplot(1,3,3);
    ft_singleplotER(cfg, GA_ERP_NC_avg, GA_ERP_CC_avg, GA_ERP_SC_avg );
    title(cfg.channel);xlabel('Time (s)');ylabel('Amplitude [uV]');
    set(gca,'fontsize',18);
    legend({'No Cue';'Center Cue';'Spatial Cue'},'location','eastoutside');
    
    
    % topoplot for P300
    cfg                 = [];
    cfg.layout = 'gu-19ch_type1';
    cfg.zlim            = [-2 2];
    cfg.xlim            = [0.4 0.7]; % P300
    cfg.style           = 'straight';
    cfg.comment         = 'no';
    cfg.marker          = 'off';
    cfg.colorbar        = 'southoutside';
    
    figure;
    subplot(1,3,1);
    ft_topoplotER(cfg, GA_ERP_2_avg);
    title('item 2');
    axis tight
    set(gca,'fontsize',14)
    
    subplot(1,3,2);
    ft_topoplotER(cfg, GA_ERP_3_avg);
    title('item 3');
    axis tight
    set(gca,'fontsize',14)
    
    subplot(1,3,3);
    ft_topoplotER(cfg, GA_ERP_4_avg);
    title('item 4');
    axis tight
    set(gca,'fontsize',14)
    
    
    %% MUA - ANOVA
    % 2-way ANOVA
    group_idx = [group_info group_info]; % valid / invalid
    cond_idx = [ones(1,subj) 2.*ones(1,subj)];
    ERP_amp_valid = zeros(1,subj);
    ERP_amp_invalid = zeros(1,subj);
    p = [];
    for ch = 1:length(GA_ERP_valid{1}.label)
        for ti = 1:length(GA_ERP_valid{1}.time)
            for si = 1:subj
                ERP_amp_valid(1,si) = GA_ERP_valid{1, si}.avg(ch,ti);
                ERP_amp_invalid(1,si) = GA_ERP_invalid{1, si}.avg(ch,ti);
            end
            ERP_amp = [ERP_amp_valid ERP_amp_invalid];
            p(ch,ti,:) = anovan(ERP_amp',{group_idx,cond_idx},'model','interaction','varnames',{'group','valid'},'display','off');
        end
    end
    p_thd = p;
    p_thd(p>.05) = 1;
    fdr = [];
    for fi = 1:3
        fdr(:,:,fi) = fdr_bh(p(:,:,fi),.05,'pdep');
    end
    p_fdr_thd = p;
    p_fdr_thd(fdr~=1) = 1;
    sig = p_fdr_thd;
    fig_label = {'group','condition','interaction'};
    switch SOA2analysis
        case '200'
            term = 80;
        case '1000'
            term = 160;
    end
    thd_map = jet;
    lin_idx = linspace(0,1,64)';
    thd_map(lin_idx>.05,:,:) = 1;
    for fi = 1:3
    figure;imagesc(sig(:,:,fi),[0 1]);xlabel('time (s)');ylabel('channel');title(fig_label{fi});
    set(gca,'xtick',1:term:length(GA_ERP_valid{1}.time),'xticklabel',GA_ERP_valid{1}.time(1:term:length(GA_ERP_valid{1}.time)),...
        'ytick',1:5:60,'yticklabel',GA_ERP_200_valid{1}.label(1:5:60),'fontsize',12);
    hold on;xline(find(GA_ERP_valid{1}.time==0),'-.k');colormap(thd_map);
    end
    
    %% MUA
    % define the parameters for the statistical comparison
    cfg = [];
    cfg.channel     = 'all';
    cfg.latency     = 'all';
    cfg.parameter   = 'avg';
    cfg.method      = 'analytic';
    cfg.statistic   = 'ft_statfun_depsamplesT';
    cfg.alpha       = 0.05;
    cfg.correctm    = 'fdr';
    
    Nsub = length(Sub_idx);
    cfg.design(1,1:2*Nsub)  = [ones(1,Nsub) 2*ones(1,Nsub)];
    cfg.design(2,1:2*Nsub)  = [1:Nsub 1:Nsub];
    cfg.ivar                = 1; % the 1st row in cfg.design contains the independent variable
    cfg.uvar                = 2; % the 2nd row in cfg.design contains the subject number
    
    stat = ft_timelockstatistics(cfg,GA_ERP_valid{Sub_idx},GA_ERP_invalid{Sub_idx});   % don't forget the {:}!
    
    GA_ERP_valid_avg.mask = stat.mask;
    cfg = [];
    cfg.maskparameter = 'mask';
    cfg.maskfacealpha = 0.5;
        cfg.fontsize = 6;
    cfg.layout = 'snuh-60ch';
    cfg.xlim = [-0.2 1.4];
%     cfg.ylim = [-6 4];
    cfg.interactive = 'yes';
    cfg.showoutline = 'yes';
    
    figure;ft_multiplotER(cfg, GA_ERP_valid_avg, GA_ERP_invalid_avg);
%     legend({'valid';'invalid';});
    set(gcf,'Position',[1 1 1239 945]);
    
    %% cluster-based permutation test - interaction effect
    if strcmp(Sub2analysis, 'All')
        Sub_con_idx = find(group_info==1);
        Sub_RBD_idx = find(group_info==2);
        for si = Sub_idx
            GA_ERP_diff{si} = GA_ERP_valid{si};
            GA_ERP_diff{si}.avg = GA_ERP_valid{si}.avg-GA_ERP_invalid{si}.avg
        end
        GA_ERP_diff_con = GA_ERP_diff(Sub_con_idx);
        GA_ERP_diff_rbd = GA_ERP_diff(Sub_RBD_idx);
        load(['Analysis_EEG\preproc\01_raw_clean.mat']);
        data = data_clean;
        cfg_neighb = [];
        cfg_neighb.layout = 'snuh-60ch';
        cfg_neighb.method = 'distance';
        cfg_neighb.elecfile = 'Z:\데이터\Posner_ERP\preprocessing\control\RBD_C01\C01_1.set';
        neighbours      = ft_prepare_neighbours(cfg_neighb, data);
        clear data;
        
        % test
        cfg = [];
        cfg.neighbours = neighbours;
        cfg.latency          = 'all';
        cfg.parameter   = 'avg';
        cfg.method           = 'montecarlo';
        % cfg.statistic        = 'ft_statfun_depsamplesT';
        cfg.statistic        = 'indepsamplesT';
        cfg.correctm         = 'cluster';
        cfg.clusterstatistic = 'maxsum';
        cfg.minnbchan        = 2;
        cfg.correcttail = 'prob';
        cfg.clusteralpha            = 0.05;
        cfg.alpha            = 0.05;
        cfg.numrandomization = 1000;
        
        Nsub = length(Sub_idx);
        cfg.design(1,1:Nsub)  = [ones(1,length(Sub_con_idx)) 2*ones(1,length(Sub_RBD_idx))];
        cfg.ivar                = 1; % the 1st row in cfg.design contains the independent variable
        
        stat_cluster = ft_timelockstatistics(cfg,GA_ERP_diff_con{:},GA_ERP_diff_rbd{:});   % don't forget the {:}!.
        
        % make a plot
        cfg = [];
        cfg.highlightsymbolseries = ['*','*','.','.','.'];
        cfg.layout = 'snuh-60ch';
        cfg.contournum = 0;
        cfg.markersymbol = '.';
        cfg.alpha = 0.05;
        cfg.parameter='stat';
        cfg.zlim = [-5 5];
        ft_clusterplot(cfg,stat_cluster);
    else
        error('all subjects are needed');
    end
    
    %% cluster-based permutation test - between IO
    % specifies with which sensors other sensors can form clusters
    load(['Analysis_EEG\preproc\01_raw_clean.mat']);
    data = data_clean;
    cfg_neighb = [];
    cfg_neighb.layout = 'snuh-60ch';
    cfg_neighb.method = 'distance';
    cfg_neighb.elecfile = 'Z:\데이터\Posner_ERP\preprocessing\control\RBD_C01\C01_1.set';
    neighbours      = ft_prepare_neighbours(cfg_neighb, data);
    clear data;
    
    % test
    cfg = [];
    cfg.neighbours = neighbours;
    cfg.latency          = 'all';
    cfg.parameter   = 'avg';
    cfg.method           = 'montecarlo';
    % cfg.statistic        = 'ft_statfun_depsamplesT';
    cfg.statistic        = 'indepsamplesT';
    cfg.correctm         = 'cluster';
    cfg.clusterstatistic = 'maxsum';
    cfg.minnbchan        = 2;
    cfg.correcttail = 'prob';
    cfg.clusteralpha            = 0.05;
    cfg.alpha            = 0.05;
    cfg.numrandomization = 1000;
    
    Nsub = length(Sub_idx);
    cfg.design(1,1:Nsub)  = [ones(1,length(Sub_con_idx)) 2*ones(1,length(Sub_RBD_idx))];
    cfg.ivar                = 1; % the 1st row in cfg.design contains the independent variable
    
    stat_cluster = ft_timelockstatistics(cfg,GA_ERP{Sub_idx});   % don't forget the {:}!.
    
    % make a plot
    cfg = [];
    cfg.highlightsymbolseries = ['*','*','.','.','.'];
    cfg.layout = 'snuh-60ch';
    cfg.contournum = 0;
    cfg.markersymbol = '.';
    cfg.alpha = 0.05;
    cfg.parameter='stat';
    cfg.zlim = [-5 5];
    ft_clusterplot(cfg,stat_cluster);
    
    %% cluster-based permutation test - within IO
    % specifies with which sensors other sensors can form clusters
    load(['Analysis_EEG\preproc\01_raw_clean.mat']);
    data = data_clean;
    cfg_neighb = [];
    cfg_neighb.layout = 'snuh-60ch';
    cfg_neighb.method = 'distance';
    cfg_neighb.elecfile = 'Z:\데이터\Posner_ERP\preprocessing\control\RBD_C01\C01_1.set';
    neighbours      = ft_prepare_neighbours(cfg_neighb, data);
    clear data;
    
    % test
    cfg = [];
    cfg.neighbours = neighbours;
    cfg.latency          = 'all';
    cfg.parameter   = 'avg';
    cfg.method           = 'montecarlo';
    % cfg.statistic        = 'ft_statfun_depsamplesT';
    cfg.statistic        = 'depsamplesT';
    cfg.correctm         = 'cluster';
    cfg.clusterstatistic = 'maxsum';
    cfg.minnbchan        = 2;
    cfg.correcttail = 'prob';
    cfg.clusteralpha            = 0.01;
    cfg.alpha            = 0.05;
    cfg.numrandomization = 1000;
    
    Nsub = length(Sub_idx);
    cfg.design(1,1:2*Nsub)  = [ones(1,Nsub) 2*ones(1,Nsub)];
    cfg.design(2,1:2*Nsub)  = [1:Nsub 1:Nsub];
    cfg.ivar                = 1; % the 1st row in cfg.design contains the independent variable
    cfg.uvar                = 2; % the 2nd row in cfg.design contains the subject number
    
    stat_cluster = ft_timelockstatistics(cfg,GA_ERP_valid{Sub_idx},GA_ERP_invalid{Sub_idx});   % don't forget the {:}!.
    
    % make a plot
    cfg = [];
    cfg.highlightsymbolseries = ['*','*','.','.','.'];
    cfg.layout = 'snuh-60ch';
    cfg.contournum = 0;
    cfg.markersymbol = '.';
    cfg.alpha = 0.05;
    cfg.parameter='stat';
    cfg.zlim = [-5 5];
    ft_clusterplot(cfg,stat_cluster);
    
    %% 2.1.2.a. visualize significant cluster ver 2.1
    % 1. plot contrast
    % 2. plot significant electrode for 70% of time interval
    cfg = [];
    cfg.sign = 'neg';
    cfg.number = 1;
    cfg.layout = 'snuh-60ch';
    cfg.threshold = 80;
    cfg.zlim = [-4 4];
    cfg.parameter = 'stat';
    cfg.comment = 'yes';
    cfg.vismode = 'topo';
    [Chan Time] = ftc_vis_cluster(cfg,stat_cluster,GA_ERP_valid,GA_ERP_invalid);
    
    % plot time series
    cfg = [];
    cfg.sign = 'neg';
    cfg.number = 1;
    cfg.layout = 'snuh-60ch';
    cfg.threshold = 80;
    cfg.zlim = [-4 4];
    cfg.parameter = 'stat';
    cfg.comment = 'yes';
    cfg.vismode = 'ts';
    cfg.xlim_ts = [-.2 .6];
    cfg.ylim_ts = [-3 3];
    ftc_vis_cluster(cfg,stat_cluster,GA_ERP_valid,GA_ERP_invalid);
    legend('valid','invalid');
    
    %% ANOVA for ROI
    Sub_idx = 1:subj; % case for ALL subject
    % 2-way ANOVA
    group_idx = [group_info group_info]; % valid / invalid
    cond_idx = [ones(1,subj) 2.*ones(1,subj)];
    
    chan_roi = Chan;
    time_roi = Time;
    ERP_amp_valid_roi = zeros(1,subj);
    ERP_amp_invalid_roi = zeros(1,subj);
    % channel, time ROI
    p = [];
    for si = 1:subj
        ERP_amp_valid_roi(1,si) = nanmean(nanmean(GA_ERP_valid{1, si}.avg(Chan,time_roi(1)<=GA_ERP_valid{1,si}.time&GA_ERP_valid{1,si}.time<time_roi(end)),1),2);
        ERP_amp_invalid_roi(1,si) = nanmean(nanmean(GA_ERP_invalid{1, si}.avg(Chan,time_roi(1)<=GA_ERP_invalid{1,si}.time&GA_ERP_invalid{1,si}.time<time_roi(end)),1),2);
    end
    ERP_amp = [ERP_amp_valid_roi ERP_amp_invalid_roi];
    p = anovan(ERP_amp',{group_idx,cond_idx},'model','interaction','varnames',{'group','valid'},'display','on');
    
    % time ROI
    ERP_amp_valid_roi = zeros(1,subj);
    ERP_amp_invalid_roi = zeros(1,subj);
    p = [];
    for ci = 1:length(GA_ERP_valid{1, si}.label)
        for si = 1:subj
            ERP_amp_valid_roi(1,si) = nanmean(nanmean(GA_ERP_valid{1, si}.avg(ci,time_roi(1)<=GA_ERP_valid{1,si}.time&GA_ERP_valid{1,si}.time<time_roi(end)),1),2);
            ERP_amp_invalid_roi(1,si) = nanmean(nanmean(GA_ERP_invalid{1, si}.avg(ci,time_roi(1)<=GA_ERP_invalid{1,si}.time&GA_ERP_invalid{1,si}.time<time_roi(end)),1),2);
        end
    end
    ERP_amp = [ERP_amp_valid_roi ERP_amp_invalid_roi];
    p = anovan(ERP_amp',{group_idx,cond_idx},'model','interaction','varnames',{'group','valid'},'display','on');
    
    
    
end

if do_cluster
%% 1. frequency analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% load the single subject averages
subj = 35;
Stat_sub = ones(1,subj);
for i = 1:subj
  details = sprintf('subject%02d', i);
  eval(details);
  load([outputdir2 filesep subjectdata.subjectnr '_' 'freq.mat'],'*_bc');
  GA_freq_200{i} = freq_200_bc;
  GA_freq_1000{i} = freq_1000_bc;
  GA_freq_200_valid{i} = freq_200_valid_bc;
  GA_freq_200_invalid{i} = freq_200_invalid_bc;
  GA_freq_1000_valid{i} = freq_1000_valid_bc;
  GA_freq_1000_invalid{i} = freq_1000_invalid_bc;
end

%% 0. visualize patterns
subj = 35;
GA_freq = [];
for i = 1:subj
    details = sprintf('subject%02d', i);
  eval(details);
load(['Analysis_EEG\Result_emotion_relchange_wavelet\' filesep subjectdata.subjectnr '_' 'freq.mat']);
cfg = [];
cfg.keeptrials    = 'no';
freq_bc_avg = ft_freqdescriptives(cfg, freq_bc);
GA_freq{i} = freq_bc_avg;
end

%% 0.1. multiplot
GA_condition = 4;
condition_label = {'200 valid', '200 invalid', '1000 valid', '1000 invalid'};
switch GA_condition
    case 1
        GA_freq = GA_freq_200_valid;
    case 2
        GA_freq = GA_freq_200_invalid;
    case 3
        GA_freq = GA_freq_1000_valid;
    case 4
        GA_freq = GA_freq_1000_invalid;
end
SOA_cond = strsplit(condition_label{GA_condition},' ');

pow = [];
for si = 1:subj
pow(:,:,:,si) = GA_freq{i}.powspctrm;
end
TFRhann = GA_freq{1};
TFRhann.powspctrm = mean(pow,4);

% cfg = [];
% cfg.showcomment = 'no';
% cfg.showlabels   = 'yes';
% cfg.showoutline = 'yes';
% cfg.layout       = 'snuh-60ch.mat';
% cfg.masknans = 'no';
% cfg.xlim = [-0.4 1];
% cfg.zlim = [-1 1];
% figure
% ft_multiplotTFR(cfg, TFRhann);title(condition_label{GA_condition});
%% 0.2. singleplot
cfg = [];	        
cfg.channel = 'all';
cfg.layout       = 'snuh-60ch.mat';
cfg.xlim = [-0.4 1];
cfg.zlim = [-1 1];
cfg.title = condition_label{GA_condition};
figure
ft_singleplotTFR(cfg, TFRhann);
xlabel('time (s)');ylabel('frequency [Hz]');
set(gca,'fontsize',12);
%% 0.3. topoplot
raweffect = TFRhann;
freq_bands = {'theta','alpha','low-beta','high-beta','gamma'};
foi_group = [4 8;9 13;13 20;20 30;30 50];
% color_maps = [-50 100;-20 40;-50 100;-50 50;-50 100];
color_maps = [-.5 .5;-.5 .5;-.5 .5;-.5 .5;-.5 .5];
% % color_maps = [0 .5;0 .5;0 .5;0 .5;0 .5];
figure;
for fi = 1:5
foi = foi_group(fi,:);
toi = [0 60];
freq_i = (foi(1)<raweffect.freq & raweffect.freq<foi(end));
time_i = (toi(1)<raweffect.time & raweffect.time<toi(end));
avg_raweffect = squeeze(nanmean(nanmean(raweffect.powspctrm(:,freq_i,time_i),3),2));
subplot(1,5,fi);topoplot(avg_raweffect,'acticap_62ch.locs','maplimits',color_maps(fi,:), 'style','map', 'electrodes','off','shading', 'interp','plotrad',.53);
title(freq_bands{fi});colormap('parula');
set(gca,'fontsize',18);
end
% %% 0. visualize patterns
% subj = 32;
% for i = 1:subj
%     details = sprintf('subject%02d', i);
%   eval(details);
% load(['Analysis_EEG\Result_emotion_relchange_mtm' filesep subjectdata.subjectnr '_' 'freq.mat']);
% GA_freq{i} = freq_bc;
% end
% raweffect = GA_freq_HV{1,1};
% foi = [8 13];
% toi = [0 120];
% freq_i = (foi(1)<raweffect.freq & raweffect.freq<foi(end));
% time_i = (toi(1)<raweffect.time & raweffect.time<toi(end));
% avg_raweffect = mean(raweffect(:,freq_i,time_i));
% 
% figure;
% topoplot(grandavg_raweffect_mask,'acticap_62ch.locs','maplimits',[-0.5 0.5], 'style','map', 'electrodes','off','shading', 'interp','plotrad',.53,...
%     'emarker2',{stat_interest_elec','x','k',10,1});


    Sub2analysis = 'All';
    switch Sub2analysis
        case 'All'
            Sub_idx = 1:subj;
        case 'CON'
            Sub_idx = find(group_info==1);
        case 'RBD'
            Sub_idx = find(group_info==2);
    end
    
    GA_freq = [];
    GA_freq_valid = [];
    GA_freq_invalid = [];
    SOA2analysis = '1000';
    switch SOA2analysis
        case '200'
            for si = 1:subj
                cfg = [];
                cfg.latency = [-0.4 1];
                GA_freq{si} = ft_selectdata(cfg, GA_freq_200{si});
                GA_freq_valid{si} = ft_selectdata(cfg, GA_freq_200_valid{si});
                GA_freq_invalid{si} = ft_selectdata(cfg, GA_freq_200_invalid{si});
            end
        case '1000'
            for si = 1:subj
                cfg = [];
                cfg.latency = [-0.4 1];
                GA_freq{si} = ft_selectdata(cfg, GA_freq_1000{si});
                GA_freq_valid{si} = ft_selectdata(cfg, GA_freq_1000_valid{si});
                GA_freq_invalid{si} = ft_selectdata(cfg, GA_freq_1000_invalid{si});
            end
    end
    

%% 1. MUA - 2-way ANOVA

    % calculate grand average for each condition
    cfg = [];
    cfg.channel   = 'all';
    cfg.latency   = 'all';
    cfg.parameter = 'powspctrm';
    GA_freq_valid_avg    = ft_freqgrandaverage(cfg,GA_freq_valid{Sub_idx});
    GA_freq_invalid_avg  = ft_freqgrandaverage(cfg,GA_freq_invalid{Sub_idx});
    
    %% MUA - ANOVA
    % 2-way ANOVA across channel
    group_idx = [group_info group_info]; % valid / invalid
    cond_idx = [ones(1,subj) 2.*ones(1,subj)];
    freq_pow_valid = zeros(1,subj);
    freq_pow_invalid = zeros(1,subj);
    p = [];
    for fi = 1:length(GA_freq_valid_avg.freq)
        for ti = 1:length(GA_freq_valid_avg.time)
            for si = 1:subj
                freq_pow_valid(1,si) = mean(GA_freq_valid{1, si}.powspctrm(:,fi,ti),1);
                freq_pow_invalid(1,si) = mean(GA_freq_invalid{1, si}.powspctrm(:,fi,ti),1);
            end
            freq_pow = [freq_pow_valid freq_pow_invalid];
            p(fi,ti,:) = anovan(freq_pow',{group_idx,cond_idx},'model','interaction','varnames',{'group','valid'},'display','off');
        end
    end
    p_thd = p;
    p_thd(p>.05) = 1;
    fdr = [];
    for fi = 1:3
        fdr(:,:,fi) = fdr_bh(p(:,:,fi),.05,'pdep');
    end
    p_fdr_thd = p;
    p_fdr_thd(fdr~=1) = 1;
    sig = p_fdr_thd;
    fig_label = {'group','condition','interaction'};
    thd_map = jet;
    lin_idx = linspace(0,1,64)';
    thd_map(lin_idx>.05,:,:) = 1;
    for fi = 1:3
    figure;imagesc(sig(:,:,fi),[0 1]);xlabel('time (s)');ylabel('frequency [Hz]');title(fig_label{fi});
    set(gca,'Ydir','normal');
    set(gca,'xtick',1:20:length(GA_freq_valid_avg.time),'xticklabel',GA_freq_valid_avg.time(1:20:length(GA_freq_valid_avg.time)),...
        'ytick',5:5:length(GA_freq_valid_avg.freq),'yticklabel',round(GA_freq_valid_avg.freq(5:5:length(GA_freq_valid_avg.freq))),'fontsize',18);
    hold on;xline(find(GA_freq_valid_avg.time==0),'-.k');colormap(thd_map);
    end
    
    
    % 2-way ANOVA
    group_idx = [group_info group_info]; % valid / invalid
    cond_idx = [ones(1,subj) 2.*ones(1,subj)];
    freq_pow_valid = zeros(1,subj);
    freq_pow_invalid = zeros(1,subj);
    p = [];
    for ch = 1:length(GA_freq_valid_avg.label)
        for fi = 1:length(GA_freq_valid_avg.freq)
            for ti = 1:length(GA_freq_valid_avg.time)
                for si = 1:subj
                    freq_pow_valid(1,si) = GA_freq_valid{1, si}.powspctrm(ch,fi,ti);
                    freq_pow_invalid(1,si) = GA_freq_invalid{1, si}.powspctrm(ch,fi,ti);
                end
                freq_pow = [freq_pow_valid freq_pow_invalid];
                p(ch,fi,ti,:) = anovan(freq_pow',{group_idx,cond_idx},'model','interaction','varnames',{'group','valid'},'display','off');
            end
        end
    end
    p_thd = p;
    p_thd(p>.05) = 1;
    fdr = [];
    for fi = 1:3
        fdr(:,:,:,fi) = fdr_bh(p(:,:,:,fi),.05,'pdep');
    end
    p_fdr_thd = p;
    p_fdr_thd(fdr~=1) = 1;
    sig = p_fdr_thd;
    fig_label = {'group','condition','interaction'};
    switch SOA2analysis
        case '200'
            term = 80;
        case '1000'
            term = 160;
    end
    thd_map = jet;
    lin_idx = linspace(0,1,64)';
    thd_map(lin_idx>.05,:,:) = 1;
    for fi = 1:3
    figure;imagesc(sig(:,:,fi),[0 1]);xlabel('time (s)');ylabel('channel');title(fig_label{fi});
    set(gca,'xtick',1:term:length(GA_ERP_valid{1}.time),'xticklabel',GA_ERP_valid{1}.time(1:term:length(GA_ERP_valid{1}.time)),...
        'ytick',1:5:60,'yticklabel',GA_ERP_200_valid{1}.label(1:5:60),'fontsize',12);
    hold on;xline(find(GA_ERP_valid{1}.time==0),'-.k');colormap(thd_map);
    end

    %% cluster-based permutation test - interaction effect
    freq_band = {'theta','alpha','beta','gamma'};
    freq_bands = [4 8;8 13;13 30;30 50];
    
    if strcmp(Sub2analysis, 'All')
        Sub_con_idx = find(group_info==1);
        Sub_RBD_idx = find(group_info==2);
        GA_freq_diff = [];
        for si = Sub_idx
            GA_freq_diff{si} = GA_freq_valid{si};
            GA_freq_diff{si}.avg = GA_freq_valid{si}.powspctrm-GA_freq_invalid{si}.powspctrm;
        end
        GA_freq_diff_con = GA_freq_diff(Sub_con_idx);
        GA_freq_diff_rbd = GA_freq_diff(Sub_RBD_idx);
        load(['Analysis_EEG\preproc\01_raw_clean.mat']);
        data = data_clean;
        cfg_neighb = [];
        cfg_neighb.layout = 'snuh-60ch';
        cfg_neighb.method = 'distance';
        cfg_neighb.elecfile = 'Z:\데이터\Posner_ERP\preprocessing\control\RBD_C01\C01_1.set';
        neighbours      = ft_prepare_neighbours(cfg_neighb, data);
        clear data;
        
        MUA_mode = 'chan_time';
        fi = 4;
        % test
        cfg = [];
        switch MUA_mode
            case 'freq_time'
                cfg.avgoverchan = 'yes';
                cfg.avgoverfreq = 'no';
            case 'chan_time'
                cfg.avgoverchan = 'no';
                cfg.avgoverfreq = 'yes';
                cfg.frequency = freq_bands(fi,:);
        end
        cfg.neighbours = neighbours;
        cfg.latency          = 'all';
        cfg.parameter   = 'powspctrm';
        cfg.method           = 'montecarlo';
        % cfg.statistic        = 'ft_statfun_depsamplesT';
        cfg.statistic        = 'indepsamplesT';
        cfg.correctm         = 'cluster';
        cfg.clusterstatistic = 'maxsum';
        cfg.minnbchan        = 2;
        cfg.correcttail = 'prob';
        cfg.clusteralpha            = 0.3;
        cfg.alpha            = 0.05;
        cfg.numrandomization = 1000;
        
        Nsub = length(Sub_idx);
        cfg.design(1,1:Nsub)  = [ones(1,length(Sub_con_idx)) 2*ones(1,length(Sub_RBD_idx))];
        cfg.ivar                = 1; % the 1st row in cfg.design contains the independent variable
        
        stat_cluster = ft_freqstatistics(cfg,GA_freq_diff_con{:},GA_freq_diff_rbd{:});   % don't forget the {:}!.
        
        % make a plot
        cfg = [];
        cfg.highlightsymbolseries = ['*','*','.','.','.'];
        cfg.layout = 'snuh-60ch';
        cfg.contournum = 0;
        cfg.markersymbol = '.';
        cfg.alpha = 0.05;
        cfg.parameter='stat';
        cfg.zlim = [-5 5];
        ft_clusterplot(cfg,stat_cluster);
    else
        error('all subjects are needed');
    end
    
    
%% 1.1. cluster-based permutation test
freq_band = {'theta','alpha','low_beta','high_beta','gamma'};
freq_bands = [4 7;8 12;14 20;20 30;30 50];
cfg = [];
cfg.latency          = 'all';
cfg.method           = 'montecarlo';
% cfg.statistic        = 'ft_statfun_depsamplesT';
cfg.statistic        = 'depsamplesT';
cfg.correctm         = 'cluster';
cfg.clusteralpha     = 0.3; temp_ca = cfg.clusteralpha;
cfg.clusterstatistic = 'maxsize';
cfg.minnbchan        = 2;
cfg.tail             = 0;
cfg.clustertail      = 0;
cfg.alpha            = 0.025; temp_a = cfg.alpha; % false alarm rate 
cfg.numrandomization = 10000;
% specifies with which sensors other sensors can form clusters
load(['raw_emotion' filesep '01_raw_clean.mat']);
data = data_clean;
data.label = {'Fp1';'Fp2';'F7';'F3';'Fz';'F4';'F8';'FC5';'FC1';'FC2';'FC6';'T7';'C3';'Cz';'C4';'T8';'TP9';'CP5';'CP1';'CP2';'CP6';'TP10';'P7';'P3';'Pz';'P4';'P8';'O1';'Oz';'O2';'AF7';'AF3';'AF4';'AF8';'F5';'F1';'F2';'F6';'FT9';'FT7';'FC3';'FC4';'FT8';'FT10';'C5';'C1';'C2';'C6';'TP7';'CP3';'CPz';'CP4';'TP8';'P5';'P1';'P2';'P6';'PO7';'PO3';'POz';'PO4';'PO8'};
cfg_neighb = [];
cfg_neighb.layout = 'acticap-62ch-standard2_BP.mat';
cfg_neighb.method = 'distance';
cfg_neighb.elecfile = 'Z:\데이터\2017_hyperscanning\Preprocessing\1_1_1_ICA.set';
cfg.neighbours       = ft_prepare_neighbours(cfg_neighb, data);
clear data;
subj = 32;
design = zeros(2,2*subj);
for i = 1:subj
    design(1,i) = i;
end
for i = 1:subj
    design(1,subj+i) = i;
end
design(2,1:subj)        = 1;
design(2,subj+1:2*subj) = 2;
cfg.design   = design;
cfg.uvar     = 1;
cfg.ivar     = 2;


%% 2.1.2 channel - time for each frequency bands
cfg.latency = 'all';
cfg.avgovertime = 'no';
cfg.avgoverfreq = 'yes';
cfg.avgoverchan = 'no';
    for i = 1:5
    cfg.frequency        = freq_bands(i,:);
    eval(['stat_V.' freq_band{i} '= ft_freqstatistics(cfg, GA_freq_HV{[1:subj]}, GA_freq_LV{[1:subj]});']);
    end
    for i = 1:5
    cfg.frequency        = freq_bands(i,:);
    eval(['stat_A.' freq_band{i} '= ft_freqstatistics(cfg, GA_freq_HA{[1:subj]}, GA_freq_LA{[1:subj]});']);
    end
    for i = 1:5
    cfg.frequency        = freq_bands(i,:);
    eval(['stat_L.' freq_band{i} '= ft_freqstatistics(cfg, GA_freq_like{[1:subj]}, GA_freq_dislike{[1:subj]});']);
    end
    
% save([outputdir filesep 'stat' filesep 'stat_V_ca' num2str(temp_ca) '_a' num2str(temp_a) '.mat'],'stat_V','temp_*');
save([outputdir2 filesep 'Results' filesep 'cluster_stat.mat'],'stat_*');

i = 5;
% visualize all significant clusters
% stat = eval(['stat_V.' freq_band{i}]);disp(['stat_V.' freq_band{i}]);
stat = eval(['stat_A.' freq_band{i}]);disp(['stat_A.' freq_band{i}]);
% stat = eval(['stat_L.' freq_band{i}]);disp(['stat_L.' freq_band{i}]);
stat.maskeffect = stat.stat;
stat.maskeffect(~stat.mask) = 0;

cfg = [];
cfg.alpha  = 0.025;
cfg.parameter = 'maskeffect';
% cfg.highlightseries = {'on','off','off','off','off'};
cfg.highlightsizeseries = [4 6 6 6 6];
cfg.zlim   = [-4 4];
cfg.layout = 'acticap-64ch-standard2_BP.mat';
cfg.visible = 'off';
% cfg.saveaspng = [outputdir filesep 'cluster_arousal_theta' filesep 'cluster'];
ft_clusterplot(cfg, stat);


%% visualization_new
% set configuration
load('chan_reset.mat'); % rearrange channel number (Front to Back, Left to Right)
% valence
climit = 2;
set_colormap = colormap(jet);
colormap_idx = linspace(-climit,climit,64);
set_colormap(-0.05<colormap_idx & colormap_idx<0.05,:) = 0.97;close;
figure;
for i = 1:4
stat = eval(['stat_V.' freq_band{i} '.stat;']);
mask = eval(['stat_V.' freq_band{i} '.mask;']);
stat(~mask) = 0;
cluster_stat = squeeze(stat);
cluster_stat= cluster_stat(chan_reset,:);
subplot(2,2,i);imagesc(cluster_stat,[-climit climit]);colormap(set_colormap);xlabel('time (s)');ylabel('channel number');
title(freq_band{i});
set(gca,'xtick',51:200:1270,'xticklabel',stat_V.theta.time(51:200:1270),'fontsize',14);
set(gca,'ytick',[13 36 53 62],'yticklabel',{'F' 'C' 'P' 'O'},'fontsize',14);
cluster_binary = cluster_stat;
cluster_binary(cluster_stat>0) = 1;
cluster_binary(cluster_stat<0) = -1;
cluster_proj(i,:) = sum(cluster_binary,1);
end
suptitle('high valence vs. low valence');
set(gcf, 'Position', [550, 150, 800, 700])
% FOR REPORT
climit = 2;
set_colormap = colormap(jet);
colormap_idx = linspace(-climit,climit,64);
set_colormap(-0.05<colormap_idx & colormap_idx<0.05,:) = 0.97;close;
figure;
i = 4;
stat = eval(['stat_V.' freq_band{i} '.stat;']);
mask = eval(['stat_V.' freq_band{i} '.mask;']);
stat(~mask) = 0;
cluster_stat = squeeze(stat);
cluster_stat= cluster_stat(chan_reset,:);
imagesc(cluster_stat,[-climit climit]);colormap(set_colormap);xlabel('time (s)');ylabel('channel number');
title([freq_band{i} '-band'],'fontsize',18);
set(gca,'xtick',51:200:1270,'xticklabel',stat_V.theta.time(51:200:1270),'fontsize',14);
set(gca,'ytick',[13 36 53 62],'yticklabel',{'F' 'C' 'P' 'O'},'fontsize',14);
set(gcf, 'Position', [550, 150, 800, 700])

% arousal
climit = 2;
set_colormap = colormap(jet);
colormap_idx = linspace(-climit,climit,62);
set_colormap(-0.05<colormap_idx & colormap_idx<0.05,:) = 0.97;close;
figure;
for i = 1:4
stat = eval(['stat_A.' freq_band{i} '.stat;']);
mask = eval(['stat_A.' freq_band{i} '.mask;']);
stat(~mask) = 0;
cluster_stat = squeeze(stat);
cluster_stat= cluster_stat(chan_reset,:);
subplot(2,2,i);imagesc(cluster_stat,[-climit climit]);colormap(set_colormap);xlabel('time (s)');ylabel('channel number');
title(freq_band{i});
set(gca,'xtick',51:200:1270,'xticklabel',stat_A.theta.time(51:200:1270),'fontsize',14);
set(gca,'ytick',[13 36 53 62],'yticklabel',{'F' 'C' 'P' 'O'},'fontsize',14);
cluster_binary = cluster_stat;
cluster_binary(cluster_stat>0) = 1;
cluster_binary(cluster_stat<0) = -1;
cluster_proj(i,:) = sum(cluster_binary,1);
end
suptitle('active vs. calm');
set(gcf, 'Position', [550, 150, 800, 700])
% FOR REPORT
climit = 2;
set_colormap = colormap(jet);
colormap_idx = linspace(-climit,climit,64);
set_colormap(-0.05<colormap_idx & colormap_idx<0.05,:) = 0.97;close;
figure;
i = 4;
stat = eval(['stat_A.' freq_band{i} '.stat;']);
mask = eval(['stat_A.' freq_band{i} '.mask;']);
stat(~mask) = 0;
cluster_stat = squeeze(stat);
cluster_stat= cluster_stat(chan_reset,:);
imagesc(cluster_stat,[-climit climit]);colormap(set_colormap);xlabel('time (s)');ylabel('channel number');
title(freq_band{i},'fontsize',18);
set(gca,'xtick',51:200:1270,'xticklabel',stat_A.theta.time(51:200:1270),'fontsize',14);
set(gca,'ytick',[13 36 53 62],'yticklabel',{'F' 'C' 'P' 'O'},'fontsize',14);
set(gcf, 'Position', [550, 150, 800, 700])
% % plot significant clusterstats distribution
% clusters = [];
% for i = 1:4
% probs = arrayfun(@(x) x.prob, eval(['stat_A.' freq_band{i} '.negclusters']))';
% clusterstats = arrayfun(@(x) x.clusterstat, eval(['stat_A.' freq_band{i} '.negclusters']))';
% temp = abs(clusterstats(probs<.025));
% clusters = [clusters;temp];
% probs = arrayfun(@(x) x.prob, eval(['stat_A.' freq_band{i} '.posclusters']))';
% clusterstats = arrayfun(@(x) x.clusterstat, eval(['stat_A.' freq_band{i} '.posclusters']))';
% temp = abs(clusterstats(probs<.025));
% clusters = [clusters;temp];
% temp = [freq_band{i} '_poscluster' num2str(j)]
% end

climit = 50;
set_colormap = colormap(jet);
colormap_idx = linspace(-climit,climit,64);
set_colormap(-1<colormap_idx & colormap_idx<1,:) = 0.97;
figure;imagesc(cluster_proj,[-climit climit]);colormap(set_colormap);xlabel('time (s)');ylabel('frequency band');
set(gca,'xtick',1:20:121,'xticklabel',stat_A.theta.time(1:20:121),'ytick',1:4,'yticklabel',{'theta','alpha','beta','gamma'},'fontsize',14);


%% 2.1.2.a. visualize significant cluster ver 2.0
% 1. plot contrast
% 2. plot significant electrode for 70% of time interval
i = 4;
stat = eval(['stat_V.' freq_band{i}]);disp(['stat_V.' freq_band{i}]);
% stat = eval(['stat_A.' freq_band{i}]);disp(['stat_A.' freq_band{i}]);
Cond1 = GA_freq_HV;
Cond2 = GA_freq_LV;
% Cond1 = GA_freq_HA;
% Cond2 = GA_freq_LA;
% Cond1 = GA_freq_like;
% Cond2 = GA_freq_dislike;
cfg = [];
cfg.foilim = freq_bands(i,:);
cfg.toilim = [80.8 87.1]; % select time of interest from cluster
grandavg_Cond1 = ft_freqgrandaverage(cfg, Cond1{:});
grandavg_Cond2 = ft_freqgrandaverage(cfg, Cond2{:});
grandavg_raweffect = squeeze(mean(mean(grandavg_Cond1.powspctrm,3),2))-squeeze(mean(mean(grandavg_Cond2.powspctrm,3),2));
time_of_interest = cfg.toilim(1)<stat.time&stat.time<cfg.toilim(2);
stat_interest = (sum(double(squeeze(stat.mask(:,:,time_of_interest))),2)/numel(find(time_of_interest)))*100;
stat_interest(stat_interest<70) = 0;
stat_interest_elec = find(logical(stat_interest));
grandavg_raweffect_mask = grandavg_raweffect;
grandavg_raweffect_mask(~ismember(1:62,stat_interest_elec)) = 0;
figure;
topoplot(grandavg_raweffect_mask,'acticap_62ch.locs','maplimits',[-0.5 0.5], 'style','map', 'electrodes','off','shading', 'interp','plotrad',.53,...
    'emarker2',{stat_interest_elec','x','k',10,1});

end % do_cluster

if do_microstate
    % fieldtrip to eeglab
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
load chan_loc_snuh_60ch.mat
EEG.chanlocs = chanlocs;

for i=1:size(data_clean.trial,2)
  EEG.data(:,:,i) = single(data_clean.trial{i}); % error here at line 17
end
EEG.setname    = 'subject01';
EEG.filename   = '';
EEG.filepath   = '';
EEG.subject    = '';
EEG.group      = '';
EEG.condition  = '';
EEG.session    = [];
EEG.comments   = 'preprocessed with fieldtrip';
EEG.nbchan     = size(data_clean.trial{1},1);
EEG.trials     = size(data_clean.trial,2);
EEG.pnts       = size(data_clean.trial{1},2);
EEG.srate      = data_clean.fsample;
EEG.xmin       = data_clean.time{1}(1);
EEG.xmax       = data_clean.time{1}(end);
EEG.times      = data_clean.time{1};
EEG.ref        = []; %'common';
EEG.event      = [];
EEG.epoch      = [];
EEG.icawinv    = [];
EEG.icasphere  = [];
EEG.icaweights = [];
EEG.icaact     = [];
EEG.saved      = 'no';
[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
eeglab redraw
pop_eegplot( EEG, 1, 1, 1);
    
end



if do_source % 2nd step
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_load = 'valence';
freq_i = 2;

%% load the single subject averages
switch data_load
    case 'valence'
        freq_band = {'beta','gamma63','gamma84'};
        subj = 32;
        Stat_sub = ones(1,subj);
        for i = 1:subj
            sub = [1:32];
            details = sprintf('subject%02d', sub(i));
            eval(details);
            load([outputdir filesep subjectdata.subjectnr '_' 'source_HV_' freq_band{freq_i} '.mat']);
            GA_source_H{i} = source;
            load([outputdir filesep subjectdata.subjectnr '_' 'source_LV_' freq_band{freq_i} '.mat']);
            GA_source_L{i} = source;
        end
    case 'arousal'
        freq_band = {'beta','gamma'};
        subj = 32;
        Stat_sub = ones(1,subj);
        for i = 1:subj
            sub = [1:32];
            details = sprintf('subject%02d', sub(i));
            eval(details);
            load([outputdir filesep subjectdata.subjectnr '_' 'source_HA_' freq_band{freq_i} '.mat']);
            GA_source_H{i} = source;
            load([outputdir filesep subjectdata.subjectnr '_' 'source_LA_' freq_band{freq_i} '.mat']);
            GA_source_L{i} = source;
        end
    case 'liking'
        freq_band = {'gamma'};
        subj = 19;
        Stat_sub = ones(1,subj);
        for i = 1:subj
            sub = [2:20];
            details = sprintf('subject%02d', sub(i));
            eval(details);
            load([outputdir filesep subjectdata.subjectnr '_' 'source_like_' freq_band{freq_i} '.mat']);
            GA_source_H{i} = source;
            load([outputdir filesep subjectdata.subjectnr '_' 'source_dislike_' freq_band{freq_i} '.mat']);
            GA_source_L{i} = source;
        end
end


% run statistics over subjects %
clear stat*

cfg=[];
cfg.dim         = GA_source_H{1}.dim;
cfg.method = 'cluster';
% cfg.correctm = 'bonferroni';
cfg.method      = 'montecarlo';
cfg.statistic   = 'ft_statfun_depsamplesT';
cfg.parameter   = 'avg.pow';
cfg.correctm    = 'cluster';
cfg.numrandomization = 1000;
cfg.clusteralpha     = 0.05;
cfg.alpha       = 0.05; % note that this only implies two-sided testing
cfg.tail        = 0;

nsubj=numel(GA_source_H);
cfg.design(1,:) = [1:nsubj 1:nsubj];
cfg.design(2,:) = [ones(1,nsubj) ones(1,nsubj)*2];
cfg.uvar        = 1; % row of design matrix that contains unit variable (in this case: subjects)
cfg.ivar        = 2; % row of design matrix that contains independent variable (the conditions)

stat = ft_sourcestatistics(cfg, GA_source_H{:}, GA_source_L{:});

% interpolate the t maps to the structural MRI of the subject %
load('standard_mri.mat');
cfg = [];
cfg.parameter = 'all';
statplot = ft_sourceinterpolate(cfg, stat, mri); 
[atlas] = ft_read_atlas('D:\Hyun\fieldtrip-20170621\template\atlas\aal\ROI_MNI_V4.nii');

% plot the t values on the MRI %
cfg = [];
cfg.atlas = atlas;
% cfg.funcolorlim   = [-5 -3];
cfg.funcolorlim   = [-4 4];
% cfg.funcolorlim   = [2 4];
    cfg.funcolormap    = 'cool';
cfg.method        = 'ortho';
cfg.funparameter  = 'stat';
cfg.maskparameter = 'mask';
ft_sourceplot(cfg, statplot);

cfg = [];
cfg.atlas = atlas;
cfg.method         = 'surface';
cfg.funparameter   = 'stat';
cfg.maskparameter  = 'mask';
    cfg.funcolormap    = 'cool';
% cfg.funcolorlim   = [-4 0];
cfg.funcolorlim   = [-4 4];
% cfg.funcolorlim   = [2 4];
%     cfg.opacitylim     = [0.0 0.1];
% cfg.opacitymap     = 'vdown';
cfg.projmethod     = 'nearest';
cfg.surffile       = 'surface_white_both.mat';
cfg.surfdownsample = 10;
cfg.camlight = 'yes';
ft_sourceplot(cfg, statplot);
view ([0 90]) % top
% view ([90 0]) % side

% save([outputdir filesep 'cluster_arousal' filesep 'statplot_arousal_beta.mat'],'statplot','-v7.3');

end % do_source


%% 1. frequency analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% load the single subject averages

subj = 20;
Stat_sub = ones(1,subj);
for i = 1:subj
  details = sprintf('subject%02d', i);
  eval(details);
  load([outputdir filesep subjectdata.subjectnr '_' 'freq_cond_avg.mat']);
  GA_freq_HV{i} = freq_HV_avg;
  GA_freq_LV{i} = freq_LV_avg;
  GA_freq_HA{i} = freq_HA_avg;
  GA_freq_LA{i} = freq_LA_avg;
  GA_freq_like{i} = freq_like_avg;
  GA_freq_dislike{i} = freq_dislike_avg;
end

%% t-statistics
freq_band = {'theta','alpha','beta','gamma'};
freq_bands = [4 8;8 13;13 30;30 50];
cfg = [];
cfg.channel     = {'EEG*'}; %now all channels
cfg.latency     = [40 100];
cfg.frequency   = 'all';
% cfg.frequency   = freq_bands(2,:);
% cfg.avgovertime = 'yes';
% cfg.avgoverfreq = 'yes';
% cfg.latency     = 'all';
% cfg.avgoverchan = 'yes';
cfg.method      = 'analytic';
cfg.statistic   = 'ft_statfun_depsamplesT';
cfg.alpha       = 0.05;
cfg.correctm    = 'fdr';
subj = 20;
design = zeros(2,2*subj);
for i = 1:subj
    design(1,i) = i;
end
for i = 1:subj
    design(1,subj+i) = i;
end
design(2,1:subj)        = 1;
design(2,subj+1:2*subj) = 2;

cfg.design   = design;
cfg.uvar     = 1;
cfg.ivar     = 2;
% %     TFR for 60-120 epoch
Stat_V = ft_freqstatistics(cfg, GA_freq_HV{[1:20]}, GA_freq_LV{[1:20]});
Stat_A = ft_freqstatistics(cfg, GA_freq_HA{[1:20]}, GA_freq_LA{[1:20]});
Stat_L = ft_freqstatistics(cfg, GA_freq_like{[1:20]}, GA_freq_dislike{[1:20]});

% visualize
cfg               = [];
cfg.marker        = 'on';
cfg.layout        = 'acticap-62ch-standard2_BP.mat';
cfg.channel       = 'EEG*';
cfg.parameter     = 'stat';  % plot the t-value 
cfg.maskparameter = 'mask';  % use the thresholded probability to mask the data
Stat_V.label = {'Fp1';'Fp2';'F7';'F3';'Fz';'F4';'F8';'FC5';'FC1';'FC2';'FC6';'T7';'C3';'Cz';'C4';'T8';'TP9';'CP5';'CP1';'CP2';'CP6';'TP10';'P7';'P3';'Pz';'P4';'P8';'O1';'Oz';'O2';'AF7';'AF3';'AF4';'AF8';'F5';'F1';'F2';'F6';'FT9';'FT7';'FC3';'FC4';'FT8';'FT10';'C5';'C1';'C2';'C6';'TP7';'CP3';'CPz';'CP4';'TP8';'P5';'P1';'P2';'P6';'PO7';'PO3';'POz';'PO4';'PO8'};
Stat_A.label = {'Fp1';'Fp2';'F7';'F3';'Fz';'F4';'F8';'FC5';'FC1';'FC2';'FC6';'T7';'C3';'Cz';'C4';'T8';'TP9';'CP5';'CP1';'CP2';'CP6';'TP10';'P7';'P3';'Pz';'P4';'P8';'O1';'Oz';'O2';'AF7';'AF3';'AF4';'AF8';'F5';'F1';'F2';'F6';'FT9';'FT7';'FC3';'FC4';'FT8';'FT10';'C5';'C1';'C2';'C6';'TP7';'CP3';'CPz';'CP4';'TP8';'P5';'P1';'P2';'P6';'PO7';'PO3';'POz';'PO4';'PO8'};
Stat_L.label = {'Fp1';'Fp2';'F7';'F3';'Fz';'F4';'F8';'FC5';'FC1';'FC2';'FC6';'T7';'C3';'Cz';'C4';'T8';'TP9';'CP5';'CP1';'CP2';'CP6';'TP10';'P7';'P3';'Pz';'P4';'P8';'O1';'Oz';'O2';'AF7';'AF3';'AF4';'AF8';'F5';'F1';'F2';'F6';'FT9';'FT7';'FC3';'FC4';'FT8';'FT10';'C5';'C1';'C2';'C6';'TP7';'CP3';'CPz';'CP4';'TP8';'P5';'P1';'P2';'P6';'PO7';'PO3';'POz';'PO4';'PO8'};
figure; ft_multiplotTFR(cfg, Stat_V);
figure; ft_multiplotTFR(cfg, Stat_A);
figure; ft_multiplotTFR(cfg, Stat_L);

% visualize over chan
cfg               = [];
cfg.marker        = 'on';
cfg.layout        = 'acticap-62ch-standard2_BP.mat';
cfg.channel       = 'all';
cfg.parameter     = 'stat';  % plot the t-value 
cfg.maskparameter = 'mask';  % use the thresholded probability to mask the data
% cfg.maskstyle     = 'saturation';
cfg.maskalpha     = 0;
cfg.zlim          = [-1 1];
figure; ft_singleplotTFR(cfg, Stat_V);
figure; ft_singleplotTFR(cfg, Stat_A);
figure; ft_singleplotTFR(cfg, Stat_L);

%% 1.3. parametric statistics (time average, freq. average)
freq_band = {'theta','alpha','beta','gamma'};
freq_bands = [4 8;8 13;13 30;30 50];
for fi = 1:4
cfg = [];
cfg.channel     = {'EEG*'}; %now all channels
cfg.latency     = [40 100];
cfg.frequency   = freq_bands(fi,:);
cfg.avgovertime = 'yes';
cfg.avgoverfreq = 'yes';
cfg.method      = 'analytic';
cfg.statistic   = 'ft_statfun_depsamplesT';
cfg.alpha       = 0.05;
cfg.correctm    = 'fdr';
subj = 20;
design = zeros(2,2*subj);
for i = 1:subj
    design(1,i) = i;
end
for i = 1:subj
    design(1,subj+i) = i;
end
design(2,1:subj)        = 1;
design(2,subj+1:2*subj) = 2;
cfg.design   = design;
cfg.uvar     = 1;
cfg.ivar     = 2;
% %     TFR for 60-120 epoch
Stat_V = ft_freqstatistics(cfg, GA_freq_HV{[1:20]}, GA_freq_LV{[1:20]});
Stat_A = ft_freqstatistics(cfg, GA_freq_HA{[1:20]}, GA_freq_LA{[1:20]});
Stat_L = ft_freqstatistics(cfg, GA_freq_like{[1:20]}, GA_freq_dislike{[1:20]});
% calculate contrast
cfg = [];
cfg.toilim = [40 100];
cfg.foilim = freq_bands(fi,:);
GA_freq_HV2 = ft_freqgrandaverage(cfg,GA_freq_HV{:});
GA_freq_LV2 = ft_freqgrandaverage(cfg,GA_freq_LV{:});
GA_freq_HA2 = ft_freqgrandaverage(cfg,GA_freq_HA{:});
GA_freq_LA2 = ft_freqgrandaverage(cfg,GA_freq_LA{:});
cfg = [];
cfg.parameter = 'powspctrm';
cfg.operation = '(x1-x2)';
TFR_diff_V      = ft_math(cfg, GA_freq_HV2, GA_freq_LV2);
TFR_diff_A      = ft_math(cfg, GA_freq_HA2, GA_freq_LA2);
% visualize over time, freq
cfg = [];
% cfg.style     = 'blank';
cfg.layout    = 'acticap-62ch-standard2_BP.mat';
cfg.highlight = 'on';
cfg.highlightchannel = find(Stat_V.mask);
cfg.comment   = 'no';
cfg.zlim = [-.5 .5];
TFR_diff_V.label = {'Fp1';'Fp2';'F7';'F3';'Fz';'F4';'F8';'FC5';'FC1';'FC2';'FC6';'T7';'C3';'Cz';'C4';'T8';'TP9';'CP5';'CP1';'CP2';'CP6';'TP10';'P7';'P3';'Pz';'P4';'P8';'O1';'Oz';'O2';'AF7';'AF3';'AF4';'AF8';'F5';'F1';'F2';'F6';'FT9';'FT7';'FC3';'FC4';'FT8';'FT10';'C5';'C1';'C2';'C6';'TP7';'CP3';'CPz';'CP4';'TP8';'P5';'P1';'P2';'P6';'PO7';'PO3';'POz';'PO4';'PO8'};
TFR_diff_A.label = {'Fp1';'Fp2';'F7';'F3';'Fz';'F4';'F8';'FC5';'FC1';'FC2';'FC6';'T7';'C3';'Cz';'C4';'T8';'TP9';'CP5';'CP1';'CP2';'CP6';'TP10';'P7';'P3';'Pz';'P4';'P8';'O1';'Oz';'O2';'AF7';'AF3';'AF4';'AF8';'F5';'F1';'F2';'F6';'FT9';'FT7';'FC3';'FC4';'FT8';'FT10';'C5';'C1';'C2';'C6';'TP7';'CP3';'CPz';'CP4';'TP8';'P5';'P1';'P2';'P6';'PO7';'PO3';'POz';'PO4';'PO8'};
figure(fi); ft_topoplotER(cfg, TFR_diff_V)
title(['parametric: significant after multiple comparison correction : ' freq_band{fi}])
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% grand average for statistics
Table_val = [F_HV(Valence_sub,:) F_LV(Valence_sub,:)];
Table_aro = [F_HA(Arousal_sub,:) F_LA(Arousal_sub,:)];
Table_like = [F_like(Liking_sub,:) F_dislike(Liking_sub,:)];
for f = 1:size(F_HV,2)
[H_val(f),P_val(f)] = ttest(F_HV(Valence_sub,f),F_LV(Valence_sub,f));
[H_aro(f),P_aro(f)] = ttest(F_HA(Arousal_sub,f),F_LA(Arousal_sub,f));
[H_like(f),P_like(f)] = ttest(F_like(Liking_sub,f),F_dislike(Liking_sub,f));
end

%% look at the analysis history
% cfg           = [];
% cfg.filename  = ['AD_diff_rmb_vs_frg.html'];
% ft_analysispipeline(cfg, diff_rmb_vs_frg);


if do_connectivity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. connectivity analysis - scalp level

%% FC strength
%% load the single subject averages

subj = 32;
Stat_sub = ones(1,subj);
label_band = {'theta','alpha','beta','gamma'};
for si = 1:subj
    details = sprintf('subject%02d', si);
    eval(details)
    load(['Analysis_EEG\Result_emotion_plv\' subjectdata.subjectnr '_ISPC_PLV_time.mat']);
    if strcmp(FC_post.HV.dimord,'rpt_chan_chan_freq')
        GAFC_post.HV.plv(:,:,:,si) = squeeze(nanmean(FC_post.HV.plvspctrm,1));
        GAFC_post.LV.plv(:,:,:,si) = squeeze(nanmean(FC_post.LV.plvspctrm,1));
        GAFC_post.HA.plv(:,:,:,si) = squeeze(nanmean(FC_post.HA.plvspctrm,1));
        GAFC_post.LA.plv(:,:,:,si) = squeeze(nanmean(FC_post.LA.plvspctrm,1));
        GAFC_post.HV.str(:,:,si) = squeeze(nanmean(FC_post.HV.plvstrength_c1,1));
        GAFC_post.LV.str(:,:,si) = squeeze(nanmean(FC_post.LV.plvstrength_c1,1));
        GAFC_post.HA.str(:,:,si) = squeeze(nanmean(FC_post.HA.plvstrength_c1,1));
        GAFC_post.LA.str(:,:,si) = squeeze(nanmean(FC_post.LA.plvstrength_c1,1));
    else
        GAFC_post.HV.plv(:,:,:,si) = FC_post.HV.plvspctrm;
        GAFC_post.LV.plv(:,:,:,si) = FC_post.LV.plvspctrm;
        GAFC_post.HA.plv(:,:,:,si) = FC_post.HA.plvspctrm;
        GAFC_post.LA.plv(:,:,:,si) = FC_post.LA.plvspctrm;
        GAFC_post.HV.str(:,:,si) = FC_post.HV.plvstrength_c1;
        GAFC_post.LV.str(:,:,si) = FC_post.LV.plvstrength_c1;
        GAFC_post.HA.str(:,:,si) = FC_post.HA.plvstrength_c1;
        GAFC_post.LA.str(:,:,si) = FC_post.LA.plvstrength_c1;
    end
end
%% High valence vs. Low valence
% strength
str_pvalue = []; str_tstat = [];
for f = 1:4
for i = 1:62
[h,p,ci,stats] = ttest(GAFC_post.HV.str(i,f,:),GAFC_post.LV.str(i,f,:));
str_pvalue(i,f) = p;
str_tstat(i,f) = stats.tstat;
H(i,f) = h;
end
end
plv_test = str_pvalue;
[h, crit_p, adj_p] = fdr_bh(plv_test,0.1);
% visualize 1
% % connectivity matrix
% figure;imagesc(h);
% visualize 2
test = nanmean(GAFC_post.HA.str,3)-nanmean(GAFC_post.LA.str,3);
test(~h) = 0;
figure;
for f = 1:4
subplot(1,4,f);topoplot(test(:,f),'acticap_62ch.locs','maplimits',[-0.5 0.5], 'style','map', 'electrodes','on','shading', 'interp','plotrad',.53,...
                'emarker2',{find(h(:,f)==1),'o','k',7,1});hold on;title(label_band{f},'fontsize',18);
end
colorbar('Position', [0.95  0.2  0.02  0.5])
set(gcf,'position',[400 400 1400 400]);

% edge
edg_pvalue = NaN(62,62,4); 
edg_tstat = NaN(62,62,4);
for f = 1:4
    for i = 1:62
        for j = i+1:62
            [h_raw(i,j,f),p,ci,stats] = ttest(GAFC_post.HV.plv(i,j,f,:),GAFC_post.LV.plv(i,j,f,:));
            edg_pvalue(i,j,f) = p;
            edg_tstat(i,j,f) = stats.tstat;
        end
    end
end
plv_test = edg_pvalue;
[h, ~, adj_p] = fdr_bh(plv_test,0.01);
% % connectivity plot
figure;
for f = 1:4
[c_col, c_row] = find(h(:,:,f));
ds = [];
ds.chanPairs = [c_col c_row];
ds.connectStrengthLimits = [-1 1];
for i = 1:size(ds.chanPairs,1)
ds.connectStrength(i,1) = edg_tstat(c_col(i),c_row(i));
end
subplot(1,4,f);topoplot_connect(ds,'acticap_62ch.locs');colormap('jet');title(label_band{f},'fontsize',18);
end
set(gcf,'position',[400 400 1400 400]);

%% High arousal vs. Low arousal
% strength
str_pvalue = []; str_tstat = [];
for f = 1:4
for i = 1:62
[h,p,ci,stats] = ttest(GAFC_post.HA.str(i,f,:),GAFC_post.LA.str(i,f,:));
str_pvalue(i,f) = p;
str_tstat(i,f) = stats.tstat;
H(i,f) = h;
end
end
plv_test = str_pvalue;
[h, crit_p, adj_p] = fdr_bh(plv_test,0.05);
% visualize 1 
% % connectivity matrix
% figure;imagesc(h);
% visualize 2
test = nanmean(GAFC_post.HA.str,3)-nanmean(GAFC_post.LA.str,3);
test(~H) = 0;
figure;
for f = 1:4
subplot(1,4,f);topoplot(test(:,f),'acticap_62ch.locs','maplimits',[-0.5 0.5], 'style','map', 'electrodes','on','shading', 'interp','plotrad',.53,...
                'emarker2',{find(H(:,f)==1),'o','k',7,1});hold on;title(label_band{f},'fontsize',18);
end
colorbar('Position', [0.95  0.2  0.02  0.5])
set(gcf,'position',[400 400 1400 400]);

% edge
edg_pvalue = NaN(62,62,4); 
edg_tstat = NaN(62,62,4);
for f = 1:4
    for i = 1:62
        for j = i+1:62
            [h,p,ci,stats] = ttest(GAFC_post.HA.plv(i,j,f,:),GAFC_post.LA.plv(i,j,f,:));
            edg_pvalue(i,j,f) = p;
            edg_tstat(i,j,f) = stats.tstat;
        end
    end
end
plv_test = edg_pvalue;
[h, crit_p, adj_p] = fdr_bh(plv_test,0.001);
% % connectivity plot
figure;
for f = 1:4
[c_col, c_row] = find(h(:,:,f)==1);
ds = [];
ds.chanPairs = [c_col c_row];
ds.connectStrengthLimits = [-1 1];
for i = 1:size(ds.chanPairs,1)
ds.connectStrength(i,1) = edg_tstat(c_col(i),c_row(i));
end
subplot(1,4,f);topoplot_connect(ds,'acticap_62ch.locs');colormap('jet');title(label_band{f},'fontsize',18);
end
set(gcf,'position',[400 400 1400 400]);

%% Like vs. Dislike
% strength
str_pvalue = []; str_tstat = [];
for f = 1:4
for i = 1:62
[h,p,ci,stats] = ttest(GAFC_post.HA.str(i,f,:),GAFC_post.LA.str(i,f,:));
str_pvalue(i,f) = p;
str_tstat(i,f) = stats.tstat;
H(i,f) = h;
end
end
plv_test = str_pvalue;
[h, crit_p, adj_p] = fdr_bh(plv_test,0.05);
% visualize 1 
% % connectivity matrix
% figure;imagesc(h);
% visualize 2
test = nanmean(GAFC_post.HA.str,3)-nanmean(GAFC_post.LA.str,3);
test(~H) = 0;
figure;
for f = 1:4
subplot(1,4,f);topoplot(test(:,f),'acticap_62ch.locs','maplimits',[-0.5 0.5], 'style','map', 'electrodes','on','shading', 'interp','plotrad',.53,...
                'emarker2',{find(H(:,f)==1),'o','k',7,1});hold on;title(label_band{f},'fontsize',18);
end
colorbar('Position', [0.95  0.2  0.02  0.5])
set(gcf,'position',[400 400 1400 400]);

% edge
edg_pvalue = NaN(62,62,4); 
edg_tstat = NaN(62,62,4);
for f = 1:4
    for i = 1:62
        for j = i+1:62
            [h,p,ci,stats] = ttest(GAFC_post.HA.plv(i,j,f,:),GAFC_post.LA.plv(i,j,f,:));
            edg_pvalue(i,j,f) = p;
            edg_tstat(i,j,f) = stats.tstat;
        end
    end
end
plv_test = edg_pvalue;
[h, crit_p, adj_p] = fdr_bh(plv_test,0.01);
% % connectivity plot
figure;
for f = 1:4
[c_col, c_row] = find(h(:,:,f));
ds = [];
ds.chanPairs = [c_col c_row];
ds.connectStrengthLimits = [-1 1];
for i = 1:size(ds.chanPairs,1)
ds.connectStrength(i,1) = edg_tstat(c_col(i),c_row(i));
end
subplot(1,4,f);topoplot_connect(ds,'acticap_62ch.locs');colormap('jet');title(label_band{f},'fontsize',18);
end
set(gcf,'position',[400 400 1400 400]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2. connectivity analysis - source level

%% FC strength
%% load the single subject averages

subj = 32;
Stat_sub = ones(1,subj);
label = {'theta','alpha','beta','gamma'};
for si = 1:subj
  details = sprintf('subject%02d', si);
  eval(details)
  load(['Analysis_EEG\Result_emotion_source_ciplv\' subjectdata.subjectnr '_Source_ciPLV_time.mat']);
  GAFC_post.HV.plv(:,:,:,si) = FC_All_post.HV.plvspctrm;
  GAFC_post.LV.plv(:,:,:,si) = FC_All_post.LV.plvspctrm;
  GAFC_post.HA.plv(:,:,:,si) = FC_All_post.HA.plvspctrm;
  GAFC_post.LA.plv(:,:,:,si) = FC_All_post.LA.plvspctrm;
  GAFC_post.HV.str(:,:,si) = FC_All_post.HV.plvstrength_c1;
  GAFC_post.LV.str(:,:,si) = FC_All_post.LV.plvstrength_c1;
  GAFC_post.HA.str(:,:,si) = FC_All_post.HA.plvstrength_c1;
  GAFC_post.LA.str(:,:,si) = FC_All_post.LA.plvstrength_c1;
end

%% High valence vs. Low valence
%% strength
str_pvalue = []; str_tstat = [];
for f = 1:4
for i = 1:62
[h,p,ci,stats] = ttest(GAFC_post.HV.str(i,f,:),GAFC_post.LV.str(i,f,:));
str_pvalue(i,f) = p;
str_tstat(i,f) = stats.tstat;
end
end
plv_test = str_pvalue;
[h, crit_p, adj_p] = fdr_bh(plv_test,0.05);
% visualize 1
% % connectivity matrix
figure;imagesc(h);
% visualize 2
  FC_label = 'GAFC_post';
  eval(['FC = ' FC_label ';'])
% node strength matrix_sig
load('default_node_strength.mat');
for f = 1:4
ds.nodeStrength = [str_tstat(:,f)];
ds.size = [1-str_pvalue(:,f)]; % 1-siginificance
node_network = [networkgenerate.x';networkgenerate.y';networkgenerate.z';ds.nodeStrength';ds.size';networkgenerate.ROI'];
fileID = fopen(['E:\2015_과제\Media\2yr\본실험\ft_pipeline\0_network_results\2017_hyperscanning_emotion\individual_valence_stat\node_brain_' FC_label '_arousal_' label{f} '.txt'],'w');
% fprintf(fileID,'%6s %12s\r\n','x','exp(x)');
fprintf(fileID,'%12.8f\t%12.8f\t%12.8f\t%12.8f\t%12.8f\t%s\r\n',node_network);
fclose(fileID);
cd E:\2015_과제\Media\2yr\본실험\ft_pipeline\0_network_results\2017_hyperscanning_emotion\individual_valence_stat\
!ren node_*.txt node_*.node
end

%% edge
edg_pvalue = NaN(62,62,4); 
edg_tstat = NaN(62,62,4);
for f = 1:4
    for i = 1:62
        for j = i+1:62
            [h_raw(i,j,f),p,ci,stats] = ttest(GAFC_post.HV.plv(i,j,f,:),GAFC_post.LV.plv(i,j,f,:));
            edg_pvalue(i,j,f) = p;
            edg_tstat(i,j,f) = stats.tstat;
        end
    end
end
plv_test = edg_pvalue;
[h, crit_p, adj_p] = fdr_bh(plv_test,0.05);
% visualize 1
figure;
for f = 1:4
    subplot(1,4,f);imagesc(h(:,:,f),[-2 2]);
end
% visualize 2, connectivity matrix_sig 0.05
h(isnan(h)) = 0;
test = nanmean(GAFC_post.HV.plv,4)-nanmean(GAFC_post.LV.plv,4);
test(~h) = 0;
  FC_label = 'GAFC_post';
  eval(['FC = ' FC_label ';'])
for f = 1:4
[c_col, c_row] = find(squeeze(h(:,:,f)));
ds = [];
ds.connectivitymat = zeros(62,62);
ds.chanPairs = [c_col c_row];
ds.connectStrengthLimits = [-1 1];
for i = 1:size(ds.chanPairs,1)
% ds.connectStrength(i,1) = plv_tstat(c_col(i),c_row(i));
ds.connectivitymat(c_col(i),c_row(i)) = test(c_col(i),c_row(i),f);
end
intra_con = ds.connectivitymat;
inter_con = zeros(62,62);
con = [intra_con inter_con;inter_con intra_con];
save(['E:\2015_과제\Media\2yr\본실험\ft_pipeline\0_network_results\2017_hyperscanning_emotion\individual_valence_stat\edge_brain_' FC_label '_valence_' label{f} '_p.txt'],'intra_con','-ascii','-double','-tabs');
cd E:\2015_과제\Media\2yr\본실험\ft_pipeline\0_network_results\2017_hyperscanning_emotion\individual_valence_stat\
!ren edge_*.txt edge_*.edge
end


%% High arousal vs. Low arousal
%% strength
str_pvalue = []; str_tstat = [];
for f = 1:4
for i = 1:62
[h,p,ci,stats] = ttest(GAFC_post.HA.str(i,f,:),GAFC_post.LA.str(i,f,:));
str_pvalue(i,f) = p;
str_tstat(i,f) = stats.tstat;
end
end
plv_test = str_pvalue;
[h, crit_p, adj_p] = fdr_bh(plv_test,0.05);
% visualize 1
% % connectivity matrix
figure;imagesc(h);

% visualize 2
  FC_label = 'GAFC_post';
  eval(['FC = ' FC_label ';'])
% node strength matrix_sig
load('default_node_strength.mat');
for f = 1:4
ds.nodeStrength = [str_tstat(:,f)];
ds.size = [1-str_pvalue(:,f)]; % 1-siginificance
node_network = [networkgenerate.x';networkgenerate.y';networkgenerate.z';ds.nodeStrength';ds.size';networkgenerate.ROI'];
fileID = fopen(['E:\2015_과제\Media\2yr\본실험\ft_pipeline\0_network_results\2017_hyperscanning_emotion\individual_arousal_stat\node_brain_' FC_label '_arousal_' label{f} '.txt'],'w');
% fprintf(fileID,'%6s %12s\r\n','x','exp(x)');
fprintf(fileID,'%12.8f\t%12.8f\t%12.8f\t%12.8f\t%12.8f\t%s\r\n',node_network);
fclose(fileID);
cd E:\2015_과제\Media\2yr\본실험\ft_pipeline\0_network_results\2017_hyperscanning_emotion\individual_arousal_stat
!ren node_*.txt node_*.node
end

%% edge
edg_pvalue = NaN(62,62,4); 
edg_tstat = NaN(62,62,4);
h_raw = NaN(62,62,4);
for f = 1:4
    for i = 1:62
        for j = i+1:62
            [h_raw(i,j,f),p,ci,stats] = ttest(GAFC_post.HA.plv(i,j,f,:),GAFC_post.LA.plv(i,j,f,:));
            edg_pvalue(i,j,f) = p;
            edg_tstat(i,j,f) = stats.tstat;
        end
    end
end
plv_test = edg_pvalue;
[h, crit_p, adj_p] = fdr_bh(plv_test,0.05);
% visualize 1
figure;
for f = 1:4
    subplot(1,4,f);imagesc(h(:,:,f));
end
% visualize 2, connectivity matrix_sig 0.05
h(isnan(h)) = 0;
test = nanmean(GAFC_post.HA.plv,4)-nanmean(GAFC_post.LA.plv,4);
test(~h) = 0;
  FC_label = 'GAFC_post';
  eval(['FC = ' FC_label ';'])
for f = 1:4
[c_col, c_row] = find(squeeze(h(:,:,f)));
ds = [];
ds.connectivitymat = zeros(62,62);
ds.chanPairs = [c_col c_row];
for i = 1:size(ds.chanPairs,1)
% ds.connectStrength(i,1) = plv_tstat(c_col(i),c_row(i));
ds.connectivitymat(c_col(i),c_row(i)) = test(c_col(i),c_row(i),f);
end
intra_con = ds.connectivitymat;
inter_con = zeros(62,62);
con = [intra_con inter_con;inter_con intra_con];
save(['E:\2015_과제\Media\2yr\본실험\ft_pipeline\0_network_results\2017_hyperscanning_emotion\individual_arousal_stat\edge_brain_' FC_label '_arousal_' label{f} '_pp.txt'],'intra_con','-ascii','-double','-tabs');
cd E:\2015_과제\Media\2yr\본실험\ft_pipeline\0_network_results\2017_hyperscanning_emotion\individual_arousal_stat\
!ren edge_*.txt edge_*.edge
end

end


if do_convergence_individual
    
    %% load behavior results
    
    load('D:\Hyun\2015_과제\Media\2yr\본실험\Result\SAM.mat');
    load('D:\Hyun\2015_과제\Media\2yr\Database\Data\Data_info.mat');
    I_HV = find(S4.Class_code==1 | S4.Class_code==2 ); % stimuli_label
    I_LV = find(S4.Class_code==3 | S4.Class_code==4 );
    I_HA = find(S4.Class_code==1 | S4.Class_code==3 );
    I_LA = find(S4.Class_code==2 | S4.Class_code==4 );
    for pi = 1:16
        delta_valence = 8-(abs(Trial_info3{pi,1}.valence-Trial_info3{pi,2}.valence));
        delta_arousal = 8-(abs(Trial_info3{pi,1}.arousal-Trial_info3{pi,2}.arousal));
        delta_V(pi,1) = mean(delta_valence);
        delta_A(pi,1) = mean(delta_arousal);
    end
    
    for i = 1:32
        if rem(i,2)
            col = 1;
        else
            col = 2;
        end
        temp = [delta_V delta_V];
        delta_V_sub(i,1) = temp(fix((i+1)/2),col);
        temp = [delta_A delta_A];
        delta_A_sub(i,1) = temp(fix((i+1)/2),col);
    end
    
    %% 1. frequency component
    %% load the single subject averages
    subj = 32;
    Stat_sub = ones(1,subj);
    for i = 1:subj
        details = sprintf('subject%02d', i);
        eval(details);
        load([outputdir2 filesep subjectdata.subjectnr '_' 'freq.mat']);
        freq_avg = ft_freqdescriptives([],freq_bc);
        GA_freq_bc{i} = freq_avg;
    end

freq_band = {'theta','alpha','low_beta','high_beta','gamma'};
freq_bands = [4 7;8 12;14 20;20 30;30 50];

GA_power = zeros(62,subj,length(freq_band));
V_freq_rho = zeros(62,length(freq_band));
V_freq_p = zeros(62,length(freq_band));
for fi = 1:length(freq_band)
cfg = [];
cfg.latency = [0 120];
cfg.frequency = freq_bands(fi,:);
for si = 1:subj
GA_freq_bc_freq{si} = ft_freqdescriptives(cfg, GA_freq_bc{si});
GA_power(:,si,fi) = squeeze(nanmean(nanmean(GA_freq_bc_freq{si}.powspctrm,2),3));
end
end
for fi = 1:length(freq_band)
for ci = 1:62
[V_freq_rho(ci,fi) V_freq_p(ci,fi)] = corr(delta_V_sub,squeeze(GA_power(ci,:,fi))','type','spearman');
[A_freq_rho(ci,fi) A_freq_p(ci,fi)] = corr(delta_A_sub,squeeze(GA_power(ci,:,fi))','type','spearman');
end
end
% inspection
V_freq_h = zeros(62,length(freq_band));
A_freq_h = zeros(62,length(freq_band));
V_conn_h2 = zeros(62,62,length(freq_band));
A_conn_h2 = zeros(62,62,length(freq_band));
V_freq_h(V_freq_p<0.05) = 1;
A_freq_h(A_freq_p<0.05) = 1;
V_conn_h2(V_freq_p<0.01) = 1;
A_conn_h2(A_freq_p<0.01) = 1;

figure;bar(freq_rho(V_freq_h(:,5)==1,5));
figure;bar(freq_rho(A_freq_h(:,4)==1,5));
% FDR correction
[V_freq_fdr, ~, ~] = fdr_bh(V_freq_p,0.05,'pdep','yes');
[A_freq_fdr, ~, ~] = fdr_bh(A_freq_p,0.05,'pdep','yes');
    
%% visualize - valence convergence
% node strength matrix_sig
load('default_node_strength.mat');
for f = 1:length(freq_band)
ds.nodeStrength = [V_freq_rho(:,f)];
ds.size = [1-V_freq_p(:,f)]; % 1-siginificance
node_network = [networkgenerate.x';networkgenerate.y';networkgenerate.z';ds.nodeStrength';ds.size';networkgenerate.ROI'];
fileID = fopen(['E:\2015_과제\Media\2yr\본실험\ft_pipeline\0_network_results\2017_hyperscanning_emotion\corr_within_eeg_power\node_brain_rho_valenceconv_' freq_band{f} '.txt'],'w');
% fprintf(fileID,'%6s %12s\r\n','x','exp(x)');
fprintf(fileID,'%12.8f\t%12.8f\t%12.8f\t%12.8f\t%12.8f\t%s\r\n',node_network);
fclose(fileID);
end
cd E:\2015_과제\Media\2yr\본실험\ft_pipeline\0_network_results\2017_hyperscanning_emotion\corr_within_eeg_power\
!ren node_*.txt node_*.node
%% visualize - arousal convergence
% node strength matrix_sig
load('default_node_strength.mat');
for f = 1:length(freq_band)
ds.nodeStrength = [A_freq_rho(:,f)];
ds.size = [1-A_freq_p(:,f)]; % 1-siginificance
node_network = [networkgenerate.x';networkgenerate.y';networkgenerate.z';ds.nodeStrength';ds.size';networkgenerate.ROI'];
fileID = fopen(['E:\2015_과제\Media\2yr\본실험\ft_pipeline\0_network_results\2017_hyperscanning_emotion\corr_within_eeg_power\node_brain_rho_arousalconv_' freq_band{f} '.txt'],'w');
% fprintf(fileID,'%6s %12s\r\n','x','exp(x)');
fprintf(fileID,'%12.8f\t%12.8f\t%12.8f\t%12.8f\t%12.8f\t%s\r\n',node_network);
fclose(fileID);
end
cd E:\2015_과제\Media\2yr\본실험\ft_pipeline\0_network_results\2017_hyperscanning_emotion\corr_within_eeg_power\
!ren node_*.txt node_*.node

    %% 2. connectivity component
    %% load the single subject averages
    freq_band = {'theta','alpha','low_beta','high_beta','gamma'};
    freq_bands = [4 7;8 12;14 20;20 30;30 50];
    
    subj = 32;
    Stat_sub = ones(1,subj);
    for si = 1:subj
        details = sprintf('subject%02d', si);
        eval(details);
        load([outputdir3 filesep subjectdata.subjectnr '_' 'Source_PLV_time.mat']);
        GAFC_post.ALL.plv(:,:,:,si) = FC_All_post.ALL.plvspctrm;
        GAFC_post.ALL.str(:,:,si) = FC_All_post.ALL.plvstrength_c1;
    end
    
    %% Strength
    V_str_pval = NaN(62,length(freq_band));
    V_str_rho = NaN(62,length(freq_band));
    A_str_pval = NaN(62,length(freq_band));
    A_str_rho = NaN(62,length(freq_band));
    for f = 1:length(freq_band)
        for i = 1:62
                [V_str_rho(i,f),V_str_pval(i,f)] = corr(delta_V_sub,squeeze(GAFC_post.ALL.str(i,f,:)),'type','spearman');
                [A_str_rho(i,f),A_str_pval(i,f)] = corr(delta_A_sub,squeeze(GAFC_post.ALL.str(i,f,:)),'type','spearman');
        end
    end
    % inspection
    V_str_h = zeros(62,length(freq_band));
    A_str_h = zeros(62,length(freq_band));
    V_str_h(V_str_pval<0.05) = 1;
    A_str_h(A_str_pval<0.05) = 1;
    for f = 1:length(freq_band)
        figure;bar(V_str_h(:,f));
    end
    % FDR correction
    [V_freq_fdr, ~, ~] = fdr_bh(V_conn_pval,0.05,'pdep','yes');
    [A_freq_fdr, ~, ~] = fdr_bh(A_conn_pval,0.05,'pdep','yes');
    
%% visualize - valence convergence
    % node strength matrix_sig
    load('default_node_strength.mat');
    for f = 1:length(freq_band)
        ds.nodeStrength = [V_str_rho(:,f)];
        ds.size = [1-V_str_pval(:,f)]; % 1-siginificance
        node_network = [networkgenerate.x';networkgenerate.y';networkgenerate.z';ds.nodeStrength';ds.size';networkgenerate.ROI'];
        fileID = fopen(['E:\2015_과제\Media\2yr\본실험\ft_pipeline\0_network_results\2017_hyperscanning_emotion\corr_within_eeg_connectivity\node_brain_rho_valenceconv_' freq_band{f} '_uncorrected.txt'],'w');
        % fprintf(fileID,'%6s %12s\r\n','x','exp(x)');
        fprintf(fileID,'%12.8f\t%12.8f\t%12.8f\t%12.8f\t%12.8f\t%s\r\n',node_network);
        fclose(fileID);
        cd E:\2015_과제\Media\2yr\본실험\ft_pipeline\0_network_results\2017_hyperscanning_emotion\corr_within_eeg_connectivity\
        !ren node_*.txt node_*.node
    end
    
%% visualize - arousal convergence
    % node strength matrix_sig
    load('default_node_strength.mat');
    for f = 1:length(freq_band)
        ds.nodeStrength = [A_str_rho(:,f)];
        ds.size = [1-A_str_pval(:,f)]; % 1-siginificance
        node_network = [networkgenerate.x';networkgenerate.y';networkgenerate.z';ds.nodeStrength';ds.size';networkgenerate.ROI'];
        fileID = fopen(['E:\2015_과제\Media\2yr\본실험\ft_pipeline\0_network_results\2017_hyperscanning_emotion\corr_within_eeg_connectivity\node_brain_rho_arousalconv_' freq_band{f} '_uncorrected.txt'],'w');
        % fprintf(fileID,'%6s %12s\r\n','x','exp(x)');
        fprintf(fileID,'%12.8f\t%12.8f\t%12.8f\t%12.8f\t%12.8f\t%s\r\n',node_network);
        fclose(fileID);
        cd E:\2015_과제\Media\2yr\본실험\ft_pipeline\0_network_results\2017_hyperscanning_emotion\corr_within_eeg_connectivity\
        !ren node_*.txt node_*.node
    end
    
    %% Edge
    V_conn_pval = NaN(62,62,length(freq_bands));
    V_conn_rho = NaN(62,62,length(freq_bands));
    A_conn_pval = NaN(62,62,length(freq_bands));
    A_conn_rho = NaN(62,62,length(freq_bands));
    for f = 1:length(freq_bands)
        for i = 1:62
            for j = i+1:62
                [V_conn_rho(i,j,f),V_conn_pval(i,j,f)] = corr(delta_V_sub,squeeze(GAFC_post.ALL.plv(i,j,f,:)),'type','spearman');
                [A_conn_rho(i,j,f),A_conn_pval(i,j,f)] = corr(delta_A_sub,squeeze(GAFC_post.ALL.plv(i,j,f,:)),'type','spearman');
            end
        end
    end
    
    % inspection
    V_conn_h = zeros(62,62,length(freq_band));
    A_conn_h = zeros(62,62,length(freq_band));
    V_conn_h2 = zeros(62,62,length(freq_band));
    A_conn_h2 = zeros(62,62,length(freq_band));
    V_conn_h3 = zeros(62,62,length(freq_band));
    A_conn_h3 = zeros(62,62,length(freq_band));
    V_conn_h(V_conn_pval<0.05) = 1;
    A_conn_h(A_conn_pval<0.05) = 1;
    V_conn_h2(V_conn_pval<0.01) = 1;
    A_conn_h2(A_conn_pval<0.01) = 1;
    V_conn_h3(V_conn_pval<0.001) = 1;
    A_conn_h3(A_conn_pval<0.001) = 1;
%     for f = 1:length(freq_band)
%         figure;imagesc(V_conn_h2(:,:,f));
%     end
    % FDR correction
    [V_freq_fdr, ~, ~] = fdr_bh(V_conn_pval,0.05,'pdep','yes');
    [A_freq_fdr, ~, ~] = fdr_bh(A_conn_pval,0.05,'pdep','yes');
    
%% visualize - valence convergence
for f = 1:length(freq_band)
[c_col, c_row] = find((squeeze(V_conn_h3(:,:,f))));
ds = [];
ds.connectivitymat = zeros(62,62);
ds.chanPairs = [c_col c_row];
ds.connectStrengthLimits = [-1 1];
for i = 1:size(ds.chanPairs,1)
ds.connectivitymat(c_col(i),c_row(i)) = V_conn_rho(c_col(i),c_row(i),f);
end
intra_con = ds.connectivitymat;
inter_con = zeros(62,62);
con = [intra_con inter_con;inter_con intra_con];
save(['E:\2015_과제\Media\2yr\본실험\ft_pipeline\0_network_results\2017_hyperscanning_emotion\corr_within_eeg_connectivity\edge_brain_rho_valenceconv_' freq_band{f} '_unppp.txt'],'intra_con','-ascii','-double','-tabs');
cd E:\2015_과제\Media\2yr\본실험\ft_pipeline\0_network_results\2017_hyperscanning_emotion\corr_within_eeg_connectivity\
!ren edge_*.txt edge_*.edge
end

%% visualize - arousal convergence
for f = 1:length(freq_band)
[c_col, c_row] = find((squeeze(A_conn_h3(:,:,f))));
ds = [];
ds.connectivitymat = zeros(62,62);
ds.chanPairs = [c_col c_row];
ds.connectStrengthLimits = [-1 1];
for i = 1:size(ds.chanPairs,1)
ds.connectivitymat(c_col(i),c_row(i)) = A_conn_rho(c_col(i),c_row(i),f);
end
intra_con = ds.connectivitymat;
inter_con = zeros(62,62);
con = [intra_con inter_con;inter_con intra_con];
save(['E:\2015_과제\Media\2yr\본실험\ft_pipeline\0_network_results\2017_hyperscanning_emotion\corr_within_eeg_connectivity\edge_brain_rho_arousalconv_' freq_band{f} '_unppp.txt'],'intra_con','-ascii','-double','-tabs');
cd E:\2015_과제\Media\2yr\본실험\ft_pipeline\0_network_results\2017_hyperscanning_emotion\corr_within_eeg_connectivity\
!ren edge_*.txt edge_*.edge
end
    
    
    
end



if do_svm
    
    %% SVM using single trial EEGs
    %% data import
    subj = 47;
    for i = 1:subj
        details = sprintf('subject%02d', i);
        eval(details);
        load([outputdir5 filesep subjectdata.subjectnr '_' 'features.mat']);
        Features{i} = Total_F;
    end
    
    R = [];
    X = [];
    for si = 1:subj
        Features_sub = [];
        for ti = 1:length(Features{si}.trial)
            Features_sub(ti,:) = Features{si}.trial{ti};
        end
        X = [X;Features_sub];
        R = [R;group_info(si)*ones(length(Features{si}.trial),1)];
    end
    
    %% classification
    % set parameters
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fsmode = 'hybrid'; % 'hybrid', 'wrapper only', 'filter only'
    train_mode = 'all_perm';
    matching = false; % # sample of class matching, true, false
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % matching # of true/false samples
    if matching
        true_idx = find(R(:,1)==1);
        false_idx = find(R(:,1)==2);
        if numel(true_idx) < numel(false_idx)
            select_idx = randi(numel(false_idx),numel(true_idx),1);
            X_s = X;
            R_V_s = R;
            select_idx = false_idx(randperm(numel(false_idx),numel(false_idx)-numel(true_idx)));
            R_V_s(select_idx,:) = [];
            X_s(select_idx,:) = [];
        elseif numel(true_idx) > numel(false_idx)
            X_s = X;
            R_V_s = R;
            select_idx = true_idx(randperm(numel(true_idx),numel(true_idx)-numel(false_idx)));
            R_V_s(select_idx,:) = [];
            X_s(select_idx,:) = [];
        end
    else
        X_s = X;
        R_V_s = R;
    end
    switch train_mode
        case 'trial'
            %         trial_list = unique(R_V_s(:,2));
            %         trial_num = length(trial_list); % # trials  training
            %         if trial_num == 0 % skip if specific channel of peri is unusable
            %             continue;
            %         end
            %         c = cvpartition(trial_num,'k',10); % 10-fold CV
        case 'all_perm'
            trial_num = length(R_V_s); % # trials  training
            c = cvpartition(length(R_V_s),'k',10); % 10-fold CV
    end
    history = cell(10,1);
    Mdl = cell(10,1);
    test_data = cell(10,1);
    test_labels = cell(10,1);
    SVM = cell(10,1);
    Conv_test = nan(10,1);
    for ci = 1:10
        switch train_mode
            case 'trial'
                %                 trIdx = trial_list(c.training(ci)); %
                %                 teIdx = trial_list(c.test(ci)); %
                %                 [Lia] = ismember(R_V_s(:,2),teIdx); % testset index
            case 'all_perm'
                trIdx = c.training(ci); %
                teIdx = c.test(ci); %
                [Lia] = teIdx; % testset index
        end
        TrFData = X_s(~Lia,:);
        TeFData = X_s(Lia,:);
        TrFR_V = R_V_s(~Lia,:);
        TeFR_V = R_V_s(Lia,:);
        %% 2. Feature selection - filter part : simple filter approach
        dataTrainG1 = TrFData(TrFR_V(:,1)==1,:);
        dataTrainG2 = TrFData(TrFR_V(:,1)==2,:);
        [h,p,~,stat] = ttest2(dataTrainG1,dataTrainG2,'Vartype','unequal');
        [~, featureIdxSortbyP] = sort(p');
        %% 3. Feature selection - Wrapper part : Forward sequential feature selection
        switch fsmode
            case 'hybrid'
                fs1 = featureIdxSortbyP(1:50); % default 150
            case 'wrapper only'
                fs1 = featureIdxSortbyP;
            case 'filter only'
                fs2 = featureIdxSortbyP(1:50); % default: 50
        end
        c_s = cvpartition(length(TrFR_V),'k',5);
        if strcmp(fsmode,'hybrid') || strcmp(fsmode,'wrapper only')
            opts = statset('display','iter');
            % 너무오래걸림
            %             classf = @(train_data,train_labels,test_data,test_labels) ...
            %                 sum(predict(fitcsvm(train_data,train_labels,'Standardize',true,'KernelFunction','rbf'),test_data)~=test_labels)/length(test_labels);
            %             [inmodel,history{ci}] = sequentialfs(classf,TrFData(:,fs1),TrFR_V(:,1),...
            %                 'cv',c_s,'options',opts,'Nf',30);
            classf = @(train_data,train_labels,test_data,test_labels) ...
                sum(predict(fitcsvm(train_data,train_labels,'Standardize',false,'Solver','L1QP'),test_data)~=test_labels)/length(test_labels);
            [inmodel,history{ci}] = sequentialfs(classf,TrFData(:,fs1),TrFR_V(:,1),...
                'cv',c_s,'options',opts,'Nf',10);
            % % plotting
            %         figure;plot(history.Crit,'o');
            %         xlabel('Number of Features');
            %         ylabel('CV MCE');
            %         title('Forward Sequential Feature Selection with cross-validation');
            [~,fs2_idx] = min(history{ci}.Crit);
            fs = fs1(history{ci}.In(fs2_idx,:));
        else
            fs = fs2;
        end
        FS{ci} = fs;
        %% 4. training model
        Mdl{ci} = fitcsvm(TrFData(:,fs),TrFR_V(:,1),'Standardize',true,'KernelFunction','rbf','OptimizeHyperparameters','auto',...
            'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
            'expected-improvement-plus','ShowPlots',false));
        %                 Mdl{ci} = fitcsvm(TrFData(:,fs),TrFR_V(:,1),'Standardize',false,'OptimizeHyperparameters','auto',...
        %                     'HyperparameterOptimizationOptions',struct('optimizer','gridsearch','ShowPlots',false),'Solver','L1QP');
        
        %% 5. regression
        Conv_test(ci,1) = Mdl{ci}.ConvergenceInfo.Converged;
        test_data{ci} = TeFData(:,fs);
        test_labels{ci} = TeFR_V(:,1);
        SVM{ci} = predict(Mdl{ci},test_data{ci});
        Accuracy(ci,1) = (sum(SVM{ci}==test_labels{ci})/length(test_labels{ci}))*100;
        Specificity(ci,1) = sum( (predict(Mdl{ci},test_data{ci}) == test_labels{ci}).*(test_labels{ci}==1) )/length(find(test_labels{ci}==1))*100;
        Sensitivity(ci,1) = sum( (predict(Mdl{ci},test_data{ci}) == test_labels{ci}).*(test_labels{ci}==2) )/length(find(test_labels{ci}==2))*100; % 2: RBD
    end
    Accuracy_avg = mean(Accuracy,1);
    Specificity_avg = mean(Specificity,1);
    Sensitivity_avg = mean(Sensitivity,1);
    save([outputdir5 '\Results\' 'Trial_EEG_' fsmode '.mat'],'history','Mdl','Conv_test','test_*','SVM','*_avg','FS','train_mode');
    
    
    
    %% SVM using Ensemble ERP and ERSP
    %% data import
    subj = 47;
    for i = 1:subj
        details = sprintf('subject%02d', i);
        eval(details);
        load([outputdir filesep subjectdata.subjectnr '_' 'timelock.mat']);
        GA_ERP{i} = ERP_All;
        GA_ERP_NC{i} = ERP_NoCue;
        GA_ERP_CC{i} = ERP_CenterCue;
        GA_ERP_SC{i} = ERP_SpatialCue;
        GA_ERP_Cg{i} = ERP_Cong_trgt;
        GA_ERP_ICg{i} = ERP_Incong_trgt;
        GA_ERP_alerting{i} = ERP_alerting;
        GA_ERP_orienting{i} = ERP_orienting;
        GA_ERP_conflict{i} = ERP_conflict;
        load([outputdir2 filesep subjectdata.subjectnr '_' 'freqlock.mat']);
        GA_ERSP{i} = ERSP_All;
        GA_ERSP_NC{i} = ERSP_NoCue;
        GA_ERSP_CC{i} = ERSP_CenterCue;
        GA_ERSP_SC{i} = ERSP_SpatialCue;
        GA_ERSP_Cg{i} = ERSP_Cong_trgt;
        GA_ERSP_ICg{i} = ERSP_Incong_trgt;
        GA_ERSP_alerting{i} = ERSP_alerting;
        GA_ERSP_orienting{i} = ERSP_orienting;
        GA_ERSP_conflict{i} = ERSP_conflict;
    end
    
    %% Feature extraction - single Ensemble ERP/ERSPs
    
    % ERP feature
    
    X_ERP = []; % Data
    X_ERP_label = []; % feature label
    
    Data = GA_ERP;
    X_temp = [];
    for si = 1:subj
        cfg = [];
        cfg.parameter = 'avg';
        cfg.windowsize = 0.05; % sec
        cfg.overlap = 0;
        ERP_F = ftc_fextraction_avgwindow(cfg, Data{si});
        X_temp(si,:) = reshape(ERP_F.winavg,1,size(ERP_F.winavg,1)*size(ERP_F.winavg,2));
    end
    label_ch = [];
    label_ch = repmat(ERP_F.label',1,length(ERP_F.wintime));
    label_t = [];
    label_t = repmat(ERP_F.wintime,1,length(ERP_F.label));
    label_temp = cell(1,length(X_temp));
    for fi = 1:length(X_temp)
        label_temp{fi} = [label_ch{fi} '_' num2str(label_t(fi))];
    end
    X_ERP = X_temp;
    X_ERP_label = label_temp;
    
    % ERSP feature
    
    X_ERSP = []; % Data
    X_ERSP_label = []; % feature label
    
    Data = GA_ERSP;
    X_temp = [];
    for si = 1:subj
        cfg = [];
        cfg.parameter = 'powspctrm';
        Data_band = ftc_fextraction_bandpower(cfg, Data{si}); % calculate band power
        ERSP = [];
        for bi = 1:length(Data_band.bandlabel)
            temp = squeeze(Data_band.bandpow(:,bi,:));
            ERSP = [ERSP reshape(temp,1,size(temp,1)*size(temp,2))];
        end
        X_temp(si,:) = ERSP;
    end
    % labeling
    label_temp2 = [];
    for bi = 1:length(Data_band.bandlabel)
        label_ch = [];
        label_ch = repmat(Data_band.label',1,length(Data_band.time));
        label_t = [];
        label_t = repmat(Data_band.time,1,length(Data_band.label));
        label_temp = cell(1,length(X_temp)/length(Data_band.bandlabel));
        for fi = 1:length(X_temp)/length(Data_band.bandlabel)
            label_temp{fi} = [Data_band.bandlabel{bi} '_' label_ch{fi} '_' num2str(label_t(fi))];
        end
        label_temp2 = [label_temp2 label_temp];
    end
    X_ERSP = X_temp;
    X_ERSP_label = label_temp2;
    
    
    %% Feature extraction - different Ensemble ERP/ERSPs
%     
%     feature_label = {'no_cue','center_cue','spatial_cue','congruent','incongruent','alerting','orienting','conflict'};
%     var_name = {'NC','CC','SC','Cg','ICg','alerting','orienting','conflict'};
%     
%     % ERP feature
%     
%     X_ERP = []; % Data
%     X_ERP_label = []; % feature label
%     
%     for vi = 1:length(var_name)
%     eval(['Data=' 'GA_ERP_' var_name{vi} ';']);
%     X_temp = [];
%     for si = 1:subj
%         cfg = [];
%         cfg.parameter = 'avg';
%         cfg.windowsize = 0.05; % sec
%         cfg.overlap = 0;
%         ERP_F = ftc_fextraction_avgwindow(cfg, Data{si});
%         X_temp(si,:) = reshape(ERP_F.winavg,1,size(ERP_F.winavg,1)*size(ERP_F.winavg,2));
%     end
%     label_ch = [];
%     label_ch = repmat(ERP_F.label',1,length(ERP_F.wintime));
%     label_t = [];
%     label_t = repmat(ERP_F.wintime,1,length(ERP_F.label));
%     label_temp = cell(1,length(X_temp));
%     for fi = 1:length(X_temp)
%         label_temp{fi} = [var_name{vi} '_' label_ch{fi} '_' num2str(label_t(fi))];
%     end
%     X_ERP = [X_ERP X_temp];
%     X_ERP_label = [X_ERP_label label_temp];
%     end
%     
%     % ERSP feature
%     
%     X_ERSP = []; % Data
%     X_ERSP_label = []; % feature label
%     
%     for vi = 1:length(var_name)
%     eval(['Data=' 'GA_ERSP_' var_name{vi} ';']);
%     X_temp = [];
%     for si = 1:subj
%         cfg = [];
%         cfg.parameter = 'powspctrm';
%         Data_band = ftc_fextraction_bandpower(cfg, Data{si}); % calculate band power
%         ERSP = [];
%         for bi = 1:length(Data_band.bandlabel)
%             temp = squeeze(Data_band.bandpow(:,bi,:));
%             ERSP = [ERSP reshape(temp,1,size(temp,1)*size(temp,2))];
%         end
%         X_temp(si,:) = ERSP;
%     end
%     % labeling
%     label_temp2 = [];
%     for bi = 1:length(Data_band.bandlabel)
%         label_ch = [];
%         label_ch = repmat(Data_band.label',1,length(Data_band.time));
%         label_t = [];
%         label_t = repmat(Data_band.time,1,length(Data_band.label));
%         label_temp = cell(1,length(X_temp)/length(Data_band.bandlabel));
%         for fi = 1:length(X_temp)/length(Data_band.bandlabel)
%             label_temp{fi} = [var_name{vi} '_' Data_band.bandlabel{bi} '_' label_ch{fi} '_' num2str(label_t(fi))];
%         end
%         label_temp2 = [label_temp2 label_temp];
%     end
%     X_ERSP = [X_ERSP X_temp];
%     X_ERSP_label = [X_ERSP_label label_temp2];
%     end
%     

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    X = [X_ERP X_ERSP];
    X_label = [X_ERP_label X_ERSP_label];
    
    % set ground-truth
    R = group_info';
    
    %% classification
    % set parameters
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    fsmode = 'hybrid'; % 'hybrid', 'wrapper only', 'filter only'
    train_mode = 'LOCV'; % LO: leave-one-out CV
    matching = false; % # sample of class matching, true, false
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % matching # of true/false samples
    if matching
        true_idx = find(R(:,1)==1);
        false_idx = find(R(:,1)==2);
        if numel(true_idx) < numel(false_idx)
            select_idx = randi(numel(false_idx),numel(true_idx),1);
            X_s = X;
            R_s = R;
            select_idx = false_idx(randperm(numel(false_idx),numel(false_idx)-numel(true_idx)));
            R_s(select_idx,:) = [];
            X_s(select_idx,:) = [];
        elseif numel(true_idx) > numel(false_idx)
            X_s = X;
            R_s = R;
            select_idx = true_idx(randperm(numel(true_idx),numel(true_idx)-numel(false_idx)));
            R_s(select_idx,:) = [];
            X_s(select_idx,:) = [];
        end
    else
        X_s = X;
        R_s = R;
    end
    switch train_mode
        case '10-Kfold'
            c = cvpartition(length(R_s),'k',10); % 10-fold CV
        case 'LOCV' % leave-one-out cross validation
            c = cvpartition(length(R_s),'LeaveOut'); % 10-fold CV            
    end
    history = cell(c.NumTestSets,1);
    Mdl = cell(c.NumTestSets,1);
    test_data = cell(c.NumTestSets,1);
    test_labels = cell(c.NumTestSets,1);
    SVM = cell(c.NumTestSets,1);
    Conv_test = nan(c.NumTestSets,1);
    for ci = 1:c.NumTestSets
        switch train_mode
            case 'LOCV'
                trIdx = c.training(ci); %
                teIdx = c.test(ci); %
                [Lia] = teIdx; % testset index
        end
        TrFData = X_s(~Lia,:);
        TeFData = X_s(Lia,:);
        TrFR_V = R_s(~Lia,:);
        TeFR_V = R_s(Lia,:);
        %% 2. Feature selection - filter part : simple filter approach
        dataTrainG1 = TrFData(TrFR_V(:,1)==1,:);
        dataTrainG2 = TrFData(TrFR_V(:,1)==2,:);
        [h,p,~,stat] = ttest2(dataTrainG1,dataTrainG2,'Vartype','unequal');
        [~, featureIdxSortbyP] = sort(p');
        %% 3. Feature selection - Wrapper part : Forward sequential feature selection
        switch fsmode
            case 'hybrid'
                fs1 = featureIdxSortbyP(1:150); % default 150
            case 'wrapper only'
                fs1 = featureIdxSortbyP;
            case 'filter only'
                fs2 = featureIdxSortbyP(1:50); % default: 50
        end
        c_s = cvpartition(length(TrFR_V),'k',5);
        if strcmp(fsmode,'hybrid') || strcmp(fsmode,'wrapper only')
            opts = statset('display','iter');
                    classf = @(train_data,train_labels,test_data,test_labels) ...
                        sum(predict(fitcsvm(train_data,train_labels,'Standardize',false,'Solver','L1QP'),test_data)~=test_labels)/length(test_labels);
                    [inmodel,history{ci}] = sequentialfs(classf,TrFData(:,fs1),TrFR_V(:,1),...
                        'cv',c_s,'options',opts,'Nf',30);
            % % plotting
            %         figure;plot(history.Crit,'o');
            %         xlabel('Number of Features');
            %         ylabel('CV MCE');
            %         title('Forward Sequential Feature Selection with cross-validation');
            [~,fs2_idx] = min(history{ci}.Crit);
            fs = fs1(history{ci}.In(fs2_idx,:));
        else
            fs = fs2;
        end
        FS{ci} = fs;
        %% 4. training model
        Mdl{ci} = fitcsvm(TrFData(:,fs),TrFR_V(:,1),'Standardize',true,'KernelFunction','rbf','OptimizeHyperparameters','auto',...
            'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
            'expected-improvement-plus','ShowPlots',false));
        %                 Mdl{ci} = fitcsvm(TrFData(:,fs),TrFR_V(:,1),'Standardize',false,'OptimizeHyperparameters','auto',...
        %                     'HyperparameterOptimizationOptions',struct('optimizer','gridsearch','ShowPlots',false),'Solver','L1QP');
        
        
        %% 5. regression
        Conv_test(ci,1) = Mdl{ci}.ConvergenceInfo.Converged;
        test_data{ci} = TeFData(:,fs);
        test_labels{ci} = TeFR_V(:,1);
        SVM{ci} = predict(Mdl{ci},test_data{ci});
        Accuracy(ci,1) = (sum(SVM{ci}==test_labels{ci})/length(test_labels{ci}))*100;
        Specificity(ci,1) = sum( (predict(Mdl{ci},test_data{ci}) == test_labels{ci}).*(test_labels{ci}==0) )/length(find(test_labels{ci}==0))*100;
        Sensitivity(ci,1) = sum( (predict(Mdl{ci},test_data{ci}) == test_labels{ci}).*(test_labels{ci}==1) )/length(find(test_labels{ci}==1))*100;
        
    end
    Accuracy_avg = mean(Accuracy,1);
    Specificity_avg = mean(Specificity,1);
    Sensitivity_avg = mean(Sensitivity,1);
    save([outputdir5 '\Results\' 'Subject_Ensemble_EEG_' fsmode '.mat'],'history','Mdl','Conv_test','test_*','SVM','*_avg','FS','train_mode');
    
    %% 6. validation 
    %% 6.1. optimal feature number
    
    train_mode = '10-Kfold';
    switch train_mode
        case '10-Kfold'
            c = cvpartition(length(R_s),'k',10); % 10-fold CV
        case 'LOCV' % leave-one-out cross validation
            c = cvpartition(length(R_s),'LeaveOut'); % 10-fold CV            
    end
    history = [];
    Mdl = [];
    test_data = [];
    test_labels = [];
    SVM = [];
    Conv_test = [];
    ci = 1;
    for fi = 1:100
        trIdx = c.training(ci); %
        teIdx = c.test(ci); %
        [Lia] = teIdx; % testset index
            TrFData = X_s(~Lia,:);
            TeFData = X_s(Lia,:);
            TrFR_V = R_s(~Lia,:);
            TeFR_V = R_s(Lia,:);
            %% 2. Feature selection - filter part : simple filter approach
            dataTrainG1 = TrFData(TrFR_V(:,1)==1,:);
            dataTrainG2 = TrFData(TrFR_V(:,1)==2,:);
            [h,p,~,stat] = ttest2(dataTrainG1,dataTrainG2,'Vartype','unequal');
            [~, featureIdxSortbyP] = sort(p');
            %% 3. Feature selection - Wrapper part : Forward sequential feature selection
            switch fsmode
                case 'hybrid'
                    fs1 = featureIdxSortbyP(1:fi); % default 150
                case 'wrapper only'
                    fs1 = featureIdxSortbyP;
                case 'filter only'
                    fs2 = featureIdxSortbyP(1:fi); % default: 50
            end
            c_s = cvpartition(length(TrFR_V),'k',5);
            if strcmp(fsmode,'hybrid') || strcmp(fsmode,'wrapper only')
                opts = statset('display','iter');
                switch Gt_mode
                    case {'valence', 'binary_valence_sam', 'arousal', 'binary_arousal_sam'}
                        %                     classf = @(train_data,train_labels,test_data,test_labels) ...
                        %                         sum(predict(fitcsvm(train_data,train_labels,'Standardize',true,'KernelFunction','rbf'),test_data)~=test_labels)/length(test_labels);
                        %                     [inmodel,history{ci}] = sequentialfs(classf,TrFData(:,fs1),TrFR_V(:,1),...
                        %                         'cv',c_s,'options',opts,'Nf',30);
                        classf = @(train_data,train_labels,test_data,test_labels) ...
                            sum(predict(fitcsvm(train_data,train_labels,'Standardize',false,'Solver','L1QP'),test_data)~=test_labels)/length(test_labels);
                        [inmodel,history{ci}] = sequentialfs(classf,TrFData(:,fs1),TrFR_V(:,1),...
                            'cv',c_s,'options',opts,'Nf',30);
                    otherwise
                        classf = @(train_data,train_labels,test_data,test_labels) ...
                            sum((predict(fitrsvm(train_data,train_labels,'Standardize',false,'KernelFunction','rbf'),test_data)-test_labels).^2)/length(test_labels);
                        [inmodel,history{ci}] = sequentialfs(classf,TrFData(:,fs1),TrFR_V(:,1),...
                            'cv',c_s,'options',opts,'Nf',30); % PartSVR: minimize resubstitution error, classf: minimize CV MSE
                end
                % % plotting
                %         figure;plot(history.Crit,'o');
                %         xlabel('Number of Features');
                %         ylabel('CV MCE');
                %         title('Forward Sequential Feature Selection with cross-validation');
                [~,fs2_idx] = min(history{ci}.Crit);
                fs = fs1(history{ci}.In(fs2_idx,:));
            else
                fs = fs2;
            end
            %% 4. training model
            Mdl = fitcsvm(TrFData(:,fs),TrFR_V(:,1),'Standardize',true,'KernelFunction','rbf');
            
            %% 5. regression
            test_data = TeFData(:,fs);
            test_labels = TeFR_V(:,1);
            SVM = predict(Mdl,test_data);
            Accuracy(fi,1) = (sum(SVM==test_labels)/length(test_labels))*100;
            Specificity(fi,1) = sum( (predict(Mdl,test_data) == test_labels).*(test_labels==0) )/length(find(test_labels==0))*100;
            Specificity(fi,1) = sum( (predict(Mdl,test_data) == test_labels).*(test_labels==1) )/length(find(test_labels==1))*100;
            waitbar(fi/500);
    end
    
    
end