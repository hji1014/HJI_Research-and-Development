%outputdir1 = 'Analysis_EEG\preproc'; % do_preprocessing
%outputdir2 = 'Analysis_EEG\erp'; % do_timelock
%outputdir3 = 'Analysis_EEG\freq'; % do_freq
outputdir1 = 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\1.H_No_cue\Control'; % do_preprocessing
outputdir2 = 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\1.H_No_cue\Control'; % do_timelock
outputdir3 = 'C:\Users\Nelab_001\Documents\MATLAB\RBD_ANT\0.HJI_sensor_analysis\1.H_No_cue\Control'; % do_freq

outputdir5 = 'Analysis_EEG\svm';
% outputdir2 = 'Analysis_EEG\Result_emotion_relchange_mtm_nosur'; % do_frequency
% outputdir3 = 'Analysis_EEG\Result_emotion_plv'; % do_connectivity
% outputdir4 = 'Analysis_EEG\Result_emotion_source'; % do_preprocessing
warning on;
do_import         = true; %
do_explore        = false; %
do_preprocessing  = true;
do_timelock       = true;
do_frequency      = true;
do_connectivity   = false;
do_connectivity_brainstorm = false;
do_stat           = false;
do_source_dics    = false;
do_source_whole  = false;
do_svm           = false;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% data import
if do_import
%   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% Reading and converting the original data files
  %%% data format conversion
  cfg.dataset             = subjectdata.eeg.datadir;
  load(cfg.dataset);
  cfg.subjectdir = subjectdata.subjectdir;
  cfg.subjectnr = subjectdata.subjectnr;
  cfg.fsample = 400;
  Data = eeg2fieldtrip(cfg,EEG_s);
  clear EEG_s
  %%% define trials
  cfg = [];
  cfg.continuous		    = 'no';
  cfg.trials              = 'all';
  cfg.offset              = 0;
  Data 			= ft_redefinetrial(cfg, Data);
end

if do_preprocessing
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% extract EEG signal
  %%% reject flunctuation for peak detection
  data_clean = Data;
%   load('chan_loc_snuh_60ch.mat');
%   cfg = [];
%   cfg.chan = chanlocs;
%   data_clean_sur = ftc_surfacelap(cfg,data_clean);
    %%% define trials
    cfg = [];
    cfg.offset              = -0.5*data_clean.fsample;
    data_clean_trgt_locked= ft_redefinetrial(cfg, data_clean);
  save([outputdir1 filesep subjectdata.subjectnr '_raw_clean.mat'],'data_clean*'); % Temporary
end % do preprocessing

if do_explore
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% Reading and reviewing the functional data
  cfg = [];
  cfg.viewmode = 'vertical';
  ft_databrowser(cfg, data_filt);
  
end % do explore

if do_timelock
    % both need the cleaned preprocessed data
    load([outputdir1 filesep subjectdata.subjectnr '_raw_clean.mat']);
    % baseline correction  & filtering
    cfg = [];
    cfg.lpfilter        = 'yes';
    cfg.lpfreq          = 30;
    cfg.demean          = 'yes';
    cfg.baselinewindow  = [-0.2 0];
    data_cue_locked       = ft_preprocessing(cfg,data_clean); % for Cue condition
    data_trgt_locked       = ft_preprocessing(cfg,data_clean_trgt_locked); % for Congruence condition
    
    cfg = [];
    cfg.trials = data_cue_locked.trialinfo.Cue==1;
    data_NoCue = ft_selectdata(cfg,data_cue_locked);
    cfg.trials = data_cue_locked.trialinfo.Cue==2;
    data_CenterCue = ft_selectdata(cfg,data_cue_locked);
    cfg.trials = data_cue_locked.trialinfo.Cue==3;
    data_SpatialCue = ft_selectdata(cfg,data_cue_locked);
    cfg = [];
    ERP_All       = ft_timelockanalysis(cfg,data_cue_locked);
    ERP_NoCue       = ft_timelockanalysis(cfg,data_NoCue);
    ERP_CenterCue       = ft_timelockanalysis(cfg,data_CenterCue);
    ERP_SpatialCue       = ft_timelockanalysis(cfg,data_SpatialCue);
    
    cfg = [];
    cfg.trials = data_trgt_locked.trialinfo.Cue==1;
    data_NoCue = ft_selectdata(cfg,data_trgt_locked);
    cfg.trials = data_trgt_locked.trialinfo.Cue==2;
    data_CenterCue = ft_selectdata(cfg,data_trgt_locked);
    cfg.trials = data_trgt_locked.trialinfo.Cue==3;
    data_SpatialCue = ft_selectdata(cfg,data_trgt_locked);
    cfg = [];
    ERP_NoCue_trgt       = ft_timelockanalysis(cfg,data_NoCue);
    ERP_CenterCue_trgt       = ft_timelockanalysis(cfg,data_CenterCue);
    ERP_SpatialCue_trgt       = ft_timelockanalysis(cfg,data_SpatialCue);
    cfg = [];
    cfg.trials = data_trgt_locked.trialinfo.Cong==1;
    data_Cong = ft_selectdata(cfg,data_trgt_locked);
    cfg.trials = data_trgt_locked.trialinfo.Cong==2;
    data_Incong = ft_selectdata(cfg,data_trgt_locked);
    cfg = [];
    ERP_Cong_trgt       = ft_timelockanalysis(cfg,data_Cong);
    ERP_Incong_trgt       = ft_timelockanalysis(cfg,data_Incong);
    
%     Ensemble average by condition
    
    ERP_Cond = cell(3,2); % Cue, Congruent
    for c1 = 1:3
        for c2 = 1:2
            cfg = [];
            cfg.trials          = find(data_trgt_locked.trialinfo.Cue == c1 & data_trgt_locked.trialinfo.Cong == c2);
            data_selected        = ft_selectdata(cfg,data_trgt_locked);
            cfg = [];
            ERP_Cond{c1,c2}       = ft_timelockanalysis(cfg,data_selected);
        end
    end
    
    cfg = [];
    cfg.operation = 'subtract';
    cfg.parameter = 'avg';
    ERP_alerting = ft_math(cfg, ERP_CenterCue, ERP_NoCue);
    ERP_orienting = ft_math(cfg, ERP_SpatialCue, ERP_CenterCue);
    ERP_conflict = ft_math(cfg, ERP_Cong_trgt, ERP_Incong_trgt);
    
    save([outputdir2 filesep subjectdata.subjectnr '_timelock.mat'],'ERP*'); % Temporary
    
    
%     cfg = [];
% %     cfg.fontsize = 6;
%     cfg.layout = 'snuh-60ch';
%         cfg.xlim = [-0.2 1.4];
%         cfg.ylim = [-6 4];
% cfg.interactive = 'yes';
% cfg.showoutline = 'yes';
%     
%     figure
%     ft_multiplotER(cfg, ERP_200, ERP_1000, ERF_SOA_diff );
%     legend({'valid';'invalid';'Difference'});
%     set(gcf,'Position',[1 1 1239 945]);
%     
%     cfg                 = [];
%     cfg.layout = 'snuh-60ch';
% %     cfg.zlim            = [-3e-13 3e-13];
%     cfg.xlim            = [0.1 0.3];
%     cfg.style           = 'straight';
%     cfg.comment         = 'no';
%     cfg.marker          = 'off';
%     cfg.colorbar        = 'southoutside';
%     
%     figure;
%     subplot(1,3,1);
%     ft_topoplotER(cfg, ERP_valid);
%     title('valid');
%     axis tight
%     
%     subplot(1,3,2);
%     ft_topoplotER(cfg, ERP_invalid);
%     title('invalid');
%     axis tight
%     
%     subplot(1,3,3);
%     ft_topoplotER(cfg, ERF_valid_diff);
%     title('Difference');
%     axis tight
%     
    
end


if do_frequency
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  %% Time-frequency analysis
    load([outputdir1 filesep subjectdata.subjectnr '_raw_clean.mat']);
  
  freq.type = 'hann'; % analysis method
  switch freq.type
      case 'wavelet' % need to adjust
%           cfg = [];
%           cfg.output = 'pow';
%           cfg.method = 'wavelet';
%           cfg.pad = 'nextpow2';
%           cfg.width = 3;
%           cfg.foi = 1:.5:50;
%           cfg.keeptrials = 'yes';
%           cfg.toi = -0.6:0.01:1.4; % adjusting parameter
%           freq_200 = ft_freqanalysis(cfg, data_clean_200);
      case 'hann'
          cfg              = [];
          cfg.output       = 'pow';
          cfg.method       = 'mtmconvol';
          cfg.pad = 'nextpow2';
          cfg.taper        = 'hanning';
          cfg.foi          = 1:1:50;                         % analysis 2 to 30 Hz in steps of 2 Hz
          cfg.t_ftimwin    = ones(length(cfg.foi),1).*0.1;   % length of time window (ex. 0.5 sec)
          cfg.keeptrials = 'yes';
          cfg.toi          = -0.4:0.05:2.2;                  % time window "slides" from -0.5 to 1.5 sec in steps of 0.05 sec (50 ms)
          freq = ft_freqanalysis(cfg, data_clean);
          cfg.toi          = -0.4:0.05:1.7;                  % time window "slides" from -0.5 to 1.5 sec in steps of 0.05 sec (50 ms)
          freq_trgt_locked = ft_freqanalysis(cfg, data_clean_trgt_locked);
  end
  %%% log power conversion for more sensitivity
%   cfg = [];
%   cfg.parameter = 'powspctrm';
%   cfg.operation = 'log10';
%   freq_log10 = ft_math(cfg,freq);
  %%% Baseline correction
  if do_baseline
    cfg           = [];
    cfg.baseline = [-0.3 -0.1];
    cfg.baselinetype = 'relchange';
    freq_bc = ft_freqbaseline(cfg,freq);
    freq_bc_trgt_locked = ft_freqbaseline(cfg,freq_trgt_locked);
  end 
  
  save(fullfile(outputdir3, [subjectdata.subjectnr '_freq']),'freq_*','-v7.3');
  
    cfg = [];
    cfg.trials = freq_bc_trgt_locked.trialinfo.Cue==1;
    freq_NoCue = ft_selectdata(cfg,freq_bc);
    cfg.trials = freq_bc_trgt_locked.trialinfo.Cue==2;
    freq_CenterCue = ft_selectdata(cfg,freq_bc);
    cfg.trials = freq_bc_trgt_locked.trialinfo.Cue==3;
    freq_SpatialCue = ft_selectdata(cfg,freq_bc);
    cfg = [];
    cfg.keeptrials = 'no';
    ERSP_All       = ft_freqdescriptives(cfg,freq_bc);
    ERSP_NoCue       = ft_freqdescriptives(cfg,freq_NoCue);
    ERSP_CenterCue       = ft_freqdescriptives(cfg,freq_CenterCue);
    ERSP_SpatialCue       = ft_freqdescriptives(cfg,freq_SpatialCue);
    
    cfg = [];
    cfg.trials = freq_bc_trgt_locked.trialinfo.Cue==1;
    freq_NoCue = ft_selectdata(cfg,freq_bc_trgt_locked);
    cfg.trials = freq_bc_trgt_locked.trialinfo.Cue==2;
    freq_CenterCue = ft_selectdata(cfg,freq_bc_trgt_locked);
    cfg.trials = freq_bc_trgt_locked.trialinfo.Cue==3;
    freq_SpatialCue = ft_selectdata(cfg,freq_bc_trgt_locked);
    cfg = [];
    cfg.keeptrials = 'no';
    ERSP_NoCue_trgt       = ft_freqdescriptives(cfg,freq_NoCue);
    ERSP_CenterCue_trgt       = ft_freqdescriptives(cfg,freq_CenterCue);
    ERSP_SpatialCue_trgt       = ft_freqdescriptives(cfg,freq_SpatialCue);
    
    cfg = [];
    cfg.trials = freq_bc_trgt_locked.trialinfo.Cong==1;
    freq_Cong = ft_selectdata(cfg,freq_bc_trgt_locked);
    cfg.trials = freq_bc_trgt_locked.trialinfo.Cong==2;
    freq_Incong = ft_selectdata(cfg,freq_bc_trgt_locked);
    cfg = [];
    ERSP_Cong_trgt       = ft_freqdescriptives(cfg,freq_Cong);
    ERSP_Incong_trgt       = ft_freqdescriptives(cfg,freq_Incong);
  
    cfg = [];
    cfg.operation = 'subtract';
    cfg.parameter = 'powspctrm';
    ERSP_alerting = ft_math(cfg, ERSP_CenterCue, ERSP_NoCue);
    ERSP_orienting = ft_math(cfg, ERSP_SpatialCue, ERSP_CenterCue);
    ERSP_conflict = ft_math(cfg, ERSP_Cong_trgt, ERSP_Incong_trgt);
    
    save([outputdir3 filesep subjectdata.subjectnr '_freqlock.mat'],'ERSP*'); % Temporary
    
  
%   % test visualization
%   cfg = [];
%   cfg.channel = 'all';
%   cfg.trials = 'all';
%   figure;ft_singleplotTFR(cfg,freq_1000);
end % do frequency

if do_connectivity
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    load([outputdir1 filesep subjectdata.subjectnr '_raw_clean.mat']);
    %%% SET conditions
    I_HV = find(data_clean.trialinfo.Stimuli_label==1 | data_clean.trialinfo.Stimuli_label==2 );
    I_LV = find(data_clean.trialinfo.Stimuli_label==3 | data_clean.trialinfo.Stimuli_label==4 );
    I_HA = find(data_clean.trialinfo.Stimuli_label==1 | data_clean.trialinfo.Stimuli_label==3 );
    I_LA = find(data_clean.trialinfo.Stimuli_label==2 | data_clean.trialinfo.Stimuli_label==4 );
    I_like = find(data_clean.trialinfo.Liking > 5 );
    I_dislike = find(data_clean.trialinfo.Liking <= 5 );
    
    % set table to array for trial information
    data_clean.trialinfo = table2array(data_clean.trialinfo);
    data_clean_sur.trialinfo = table2array(data_clean_sur.trialinfo);
    
    cfg = [];
    cfg.latency = [-14 0];
    data_clean_sur_base = ft_selectdata(cfg,data_clean_sur);
    cfg.trials = I_HV;
    data_clean_sur_base_HV = ft_selectdata(cfg,data_clean_sur);
    cfg.trials = I_LV;
    data_clean_sur_base_LV = ft_selectdata(cfg,data_clean_sur);
    cfg.trials = I_HA;
    data_clean_sur_base_HA = ft_selectdata(cfg,data_clean_sur);
    cfg.trials = I_LA;
    data_clean_sur_base_LA = ft_selectdata(cfg,data_clean_sur);
    cfg = [];
    cfg.latency = [0 120];
    data_clean_sur_post = ft_selectdata(cfg,data_clean_sur);
    cfg.trials = I_HV;
    data_clean_sur_post_HV = ft_selectdata(cfg,data_clean_sur);
    cfg.trials = I_LV;
    data_clean_sur_post_LV = ft_selectdata(cfg,data_clean_sur);
    cfg.trials = I_HA;
    data_clean_sur_post_HA = ft_selectdata(cfg,data_clean_sur);
    cfg.trials = I_LA;
    data_clean_sur_post_LA = ft_selectdata(cfg,data_clean_sur);
    
    % calculate ISPC
    freq_band = {'theta' 'alpha' 'low beta' 'high beta' 'gamma'};
    freq_foi = [6 10 17 25 40];
    freq_tapsmofreq = [2 2 3 5 10];
    freq_f = [4 8;8 12;14 20;20 30;30 50];
    cfg = [];
    cfg.foi = freq_foi;
    cfg.tapsmofreq = freq_tapsmofreq;
    cfg.method = 'PLV_time';
    cfg.segment_time = 5; % 5 sec window
    cfg.symmetric = 'no';
    cfg.baseline = 'yes';
    [FC_post.ALL, FC_base.ALL, FC_postbc.ALL] = ftc_ISPC_time(cfg, data_clean_sur_post, data_clean_sur_base);
    [FC_post.HV, FC_base.HV, FC_postbc.HV] = ftc_ISPC_time(cfg, data_clean_sur_post_HV, data_clean_sur_base_HV);
    [FC_post.LV, FC_base.LV, FC_postbc.LV] = ftc_ISPC_time(cfg, data_clean_sur_post_LV, data_clean_sur_base_LV);
    [FC_post.HA, FC_base.HA, FC_postbc.HA] = ftc_ISPC_time(cfg, data_clean_sur_post_HA, data_clean_sur_base_HA);
    [FC_post.LA, FC_base.LA, FC_postbc.LA] = ftc_ISPC_time(cfg, data_clean_sur_post_LA, data_clean_sur_base_LA);
    save(fullfile(outputdir3, [subjectdata.subjectnr '_ISPC_' cfg.method]),'FC_*');
end % do_connectivity



if do_connectivity_brainstorm
    % load data
    
  %% Reading and converting the original data files
  % Set conditions
  load([outputdir1 filesep subjectdata.subjectnr '_raw_clean.mat'],'data_clean');
  I_HV = find(data_clean.trialinfo.Stimuli_label==1 | data_clean.trialinfo.Stimuli_label==2 );
  I_LV = find(data_clean.trialinfo.Stimuli_label==3 | data_clean.trialinfo.Stimuli_label==4 );
  I_HA = find(data_clean.trialinfo.Stimuli_label==1 | data_clean.trialinfo.Stimuli_label==3 );
  I_LA = find(data_clean.trialinfo.Stimuli_label==2 | data_clean.trialinfo.Stimuli_label==4 );
  I_like = find(data_clean.trialinfo.Liking > 5 );
  I_dislike = find(data_clean.trialinfo.Liking <= 5 );
  
%   clearvars -except do* sub* outputdir Stim_i Subjects pipe_mode
clearvars *_post *_bc
  %%% data format conversion
  cfg.subjectdir = subjectdata.subjectdir;
  cfg.subjectnr = subjectdata.subjectnr;
  cfg.stim = Stim_i;
  cfg.fsample = 200;
  data = brainstorm2fieldtrip(cfg);
  %%% define trials
  cfg = [];
  cfg.continuous		    = 'no';
  cfg.trials              = 'all';
  cfg.offset              = 0;
  cfg.trialdef.eventtype	= Stim_i; % VD
  data 			= ft_redefinetrial(cfg, data);
  cfg = [];
  cfg.demean = 'yes';
  data_clean = ft_preprocessing(cfg,data);
  
  % sorting 32 trials
%   data = data1.trial;
%   Temp = data1.trial;
%   DeleteIdx = zeros(1,length(data));
%   for ti = 1:64
%       for di = ti+1:64
%           if prod(prod((Temp{1,ti}==data{1,di})))
%               DeleteIdx(di) = 1;
%           end
%       end
%   end
%   save('brainstorm_emotion_trial_overlap_idx.mat','DeleteIdx');
% load('brainstorm_emotion_trial_overlap_idx.mat');
%   data_all = data_clean;
%   data_all.trial(find(DeleteIdx)) = [];
%   data_all.time(find(DeleteIdx)) = [];
%   data_all.trialinfo(find(DeleteIdx)) = [];
%   data_all.sampleinfo(33:64,:) = [];
  
  %% no condition
  freq_band = {'theta' 'alpha' 'low beta' 'high beta' 'gamma'};
  freq_foi = [6 10 17 25 40];
  freq_tapsmofreq = [2 2 3 5 10];
  freq_f = [4 8;8 12;14 20;20 30;30 50];
  cfg           = [];
  cfg.latency   = [-14 0];
  data_base = ft_selectdata(cfg,data_clean);
  cfg.trials    = strcmp(data_clean.trialinfo,'HV');
  data_base_HV = ft_selectdata(cfg,data_clean);
  cfg.trials    = strcmp(data_clean.trialinfo,'LV');
  data_base_LV = ft_selectdata(cfg,data_clean);
  cfg.trials    = strcmp(data_clean.trialinfo,'HA');
  data_base_HA = ft_selectdata(cfg,data_clean);
  cfg.trials    = strcmp(data_clean.trialinfo,'LA');
  data_base_LA = ft_selectdata(cfg,data_clean);
  cfg.latency   = [0 120];
  data_post = ft_selectdata(cfg,data_clean);
  cfg.trials    = strcmp(data_clean.trialinfo,'HV');
  data_post_HV = ft_selectdata(cfg,data_clean);
  cfg.trials    = strcmp(data_clean.trialinfo,'LV');
  data_post_LV = ft_selectdata(cfg,data_clean);
  cfg.trials    = strcmp(data_clean.trialinfo,'HA');
  data_post_HA = ft_selectdata(cfg,data_clean);
  cfg.trials    = strcmp(data_clean.trialinfo,'LA');
  data_post_LA = ft_selectdata(cfg,data_clean);
%   
%   freq_band = {'theta' 'alpha' 'beta' 'gamma'};
%   freq_foi = [6 10 21 40];
%   freq_tapsmofreq = [2 2 9 10];
%   freq_f = [4 8;8 13;13 30;30 50];
%   cfg           = [];
%   cfg.latency   = [-14 0];
%   cfg.trials    = I_HV;
%   data_base_HV = ft_selectdata(cfg,data_all);
%   cfg.trials    = I_LV;
%   data_base_LV = ft_selectdata(cfg,data_all);
%   cfg.trials    = I_HA;
%   data_base_HA = ft_selectdata(cfg,data_all);
%   cfg.trials    = I_LA;
%   data_base_LA = ft_selectdata(cfg,data_all);
%   cfg.latency   = [0 120];
%   cfg.trials    = I_HV;
%   data_post_HV = ft_selectdata(cfg,data_all);
%   cfg.trials    = I_LV;
%   data_post_LV = ft_selectdata(cfg,data_all);
%   cfg.trials    = I_HA;
%   data_post_HA = ft_selectdata(cfg,data_all);
%   cfg.trials    = I_LA;
%   data_post_LA = ft_selectdata(cfg,data_all);
%   % calculate PLV
  cfg = [];
  cfg.foi = freq_foi;
  cfg.tapsmofreq = freq_tapsmofreq;
  cfg.method = 'PLV_time';
  cfg.segment_time = 5; % 5 sec window
  cfg.symmetric = 'no';
  cfg.baseline = 'yes';
  [FC_All_post.ALL, FC_All_base.ALL, FC_All_post_bc.ALL] = ftc_ISPC_time(cfg, data_post, data_base);
  [FC_All_post.HV, FC_All_base.HV, FC_All_post_bc.HV] = ftc_ISPC_time(cfg, data_post_HV, data_base_HV);
  [FC_All_post.LV, FC_All_base.LV, FC_All_post_bc.LV] = ftc_ISPC_time(cfg, data_post_LV, data_base_LV);
  [FC_All_post.HA, FC_All_base.HA, FC_All_post_bc.HA] = ftc_ISPC_time(cfg, data_post_HA, data_base_HA);
  [FC_All_post.LA, FC_All_base.LA, FC_All_post_bc.LA] = ftc_ISPC_time(cfg, data_post_LA, data_base_LA);
      

  if ~strcmp(pipe_mode,'surrogate')
      save(['Analysis_EEG\Result_emotion_source_plv\' subjectdata.subjectnr '_Source_' cfg.method],'FC_All_*');
  else
%       FC_HA.post{pi,bi} = FC_HA_post;
%       FC_HA.bc{pi,bi} = FC_HA_bc;
%       FC_LA.post{pi,bi} = FC_LA_post;
%       FC_LA.bc{pi,bi} = FC_LA_bc;
%       FC_HV.post{pi,bi} = FC_HV_post;
%       FC_HV.bc{pi,bi} = FC_HV_bc;
%       FC_LV.post{pi,bi} = FC_LV_post;
%       FC_LV.bc{pi,bi} = FC_LV_bc;
%       FC_A.post{pi,bi} = FC_A_post;
%       FC_A.base{pi,bi} = FC_A_base;
  end
end


if do_stat
    %% Cluster-based permutation test 
    clearvars -except do* subj subjectdata outputdir Stim_i Subjects
    load([outputdir filesep subjectdata.subjectnr '_raw_clean.mat']);
    load([outputdir filesep subjectdata.subjectnr '_' Stim_i '_' 'freq_HV.mat']);
    freq_HV = freq;
    load([outputdir filesep subjectdata.subjectnr '_' Stim_i '_' 'freq_LV.mat']);
    freq_LV = freq;
    load([outputdir filesep subjectdata.subjectnr '_' Stim_i '_' 'freq_HA.mat']);
    freq_HA = freq;
    load([outputdir filesep subjectdata.subjectnr '_' Stim_i '_' 'freq_LA.mat']);
    freq_LA = freq;
    load([outputdir filesep subjectdata.subjectnr '_' Stim_i '_' 'freq_like.mat']);
    freq_like = freq;
    load([outputdir filesep subjectdata.subjectnr '_' Stim_i '_' 'freq_dislike.mat']);
    freq_dislike = freq;
    freq_band = {'theta','alpha','beta','gamma'};
    freq_bands = [4 8;8 13;13 30;30 50];
    cfg = [];
    cfg.channel          = 'EEG*';
    cfg.latency          = 'all';
    cfg.method           = 'montecarlo';
    cfg.statistic        = 'ft_statfun_indepsamplesT';
    cfg.correctm         = 'cluster';
    cfg.clusteralpha     = 0.05;
    cfg.clusterstatistic = 'maxsum';
    cfg.minnbchan        = 2;
    cfg.tail             = 0;
    cfg.clustertail      = 0;
    cfg.alpha            = 0.025;
    cfg.numrandomization = 500;
    % prepare_neighbours determines what sensors may form clusters
    data = data_clean_sur;
    data.label = {'Fp1';'Fp2';'F7';'F3';'Fz';'F4';'F8';'FC5';'FC1';'FC2';'FC6';'T7';'C3';'Cz';'C4';'T8';'TP9';'CP5';'CP1';'CP2';'CP6';'TP10';'P7';'P3';'Pz';'P4';'P8';'O1';'Oz';'O2';'AF7';'AF3';'AF4';'AF8';'F5';'F1';'F2';'F6';'FT9';'FT7';'FC3';'FC4';'FT8';'FT10';'C5';'C1';'C2';'C6';'TP7';'CP3';'CPz';'CP4';'TP8';'P5';'P1';'P2';'P6';'PO7';'PO3';'POz';'PO4';'PO8'};
    cfg_neighb = [];
    cfg_neighb.layout = 'acticap-62ch-standard2_BP.mat';
    cfg_neighb.method = 'distance';
    cfg_neighb.elecfile = 'D:\Hyun\2015_과제\Media\2yr\Database\Data\Preproc\ICA\1_2_2_ICA.set';
    cfg.neighbours       = ft_prepare_neighbours(cfg_neighb, data);
    clear data;
    design = zeros(1,size(freq_HV.powspctrm,1) + size(freq_LV.powspctrm,1));
    design(1,1:size(freq_HV.powspctrm,1)) = 1;
    design(1,(size(freq_HV.powspctrm,1)+1):(size(freq_HV.powspctrm,1)+...
        size(freq_LV.powspctrm,1))) = 2;
    cfg.design           = design;
    cfg.ivar             = 1;
    cfg.avgoverfreq      = 'yes';
    for i = 1:4
    cfg.frequency        = freq_bands(i,:);
    eval(['stat_V.' freq_band{i} '= ft_freqstatistics(cfg, freq_HV, freq_LV);']);
    end
    for i = 1:4
    cfg.frequency        = freq_bands(i,:);
    eval(['stat_A.' freq_band{i} '= ft_freqstatistics(cfg, freq_HV, freq_LV);']);
    end
    for i = 1:4
    cfg.frequency        = freq_bands(i,:);
    eval(['stat_L.' freq_band{i} '= ft_freqstatistics(cfg, freq_HV, freq_LV);']);
    end
    save([outputdir filesep subjectdata.subjectnr '_stat_perm.mat'],'stat_V');
    save([outputdir filesep subjectdata.subjectnr '_stat_perm.mat'],'stat_A','-append');
    save([outputdir filesep subjectdata.subjectnr '_stat_perm.mat'],'stat_L','-append');
    % Plotting the results
    cfg = [];
    for i = 1:4
    cfg.frequency = freq_bands(i,:);
    freq_HV_avg = ft_freqdescriptives(cfg, freq_HV);
    freq_LV_avg  = ft_freqdescriptives(cfg, freq_LV);
    eval(['stat_V.' freq_band{i} '.raweffect = mean(freq_HV_avg.powspctrm,2) - mean(freq_LV_avg.powspctrm,2);']);
    freq_HA_avg = ft_freqdescriptives(cfg, freq_HA);
    freq_LA_avg  = ft_freqdescriptives(cfg, freq_LA);
    eval(['stat_A.' freq_band{i} '.raweffect = mean(freq_HA_avg.powspctrm,2) - mean(freq_LA_avg.powspctrm,2);']);
    freq_like_avg = ft_freqdescriptives(cfg, freq_like);
    freq_dislike_avg  = ft_freqdescriptives(cfg, freq_dislike);
    eval(['stat_L.' freq_band{i} '.raweffect = mean(freq_like_avg.powspctrm,2) - mean(freq_dislike_avg.powspctrm,2);']);
    end
%     cfg = [];
%     cfg.alpha  = 0.025;
%     cfg.parameter = 'raweffect';
%     cfg.zlim   = [-1e-27 1e-27];
%     cfg.layout = 'acticap-62ch-standard2_BP.mat';
%     for i = 1:4
%         eval(['ft_clusterplot(cfg, stat_V.' freq_band{i} ');']);
%     end
%     for i = 1:4
%         eval(['ft_clusterplot(cfg, stat_A.' freq_band{i} ');']);
%     end
%     for i = 1:4
%         eval(['ft_clusterplot(cfg, stat_L.' freq_band{i} ');']);
%     end
    
end % do stat


if do_source_dics
   
    %% 1 Reading the cleaned data
    load([outputdir1 filesep subjectdata.subjectnr '_raw_clean.mat']);
    data_clean.trialinfo = table2array(data_clean.trialinfo);
    % condition
    trialinfo = data_clean.trialinfo;
    %%% SET conditions
    I_HV = find(data_clean.trialinfo(:,3)==1 | data_clean.trialinfo(:,3)==2 );
    I_LV = find(data_clean.trialinfo(:,3)==3 | data_clean.trialinfo(:,3)==4);
    I_HA = find(data_clean.trialinfo(:,3)==1 | data_clean.trialinfo(:,3)==3);
    I_LA = find(data_clean.trialinfo(:,3)==2 | data_clean.trialinfo(:,3)==4);
    I_like = find(data_clean.trialinfo(:,6)>5);
    I_dislike = find(data_clean.trialinfo(:,6)<=5);
%     
%    %% 1. valence
   
    %%%%%%%%%%% valence configurations
    freq_label = {'beta','gamma63','gamma84'};
    freq_toi = zeros(length(freq_label),2);
    freq_toi(1,:) = [62 66.8]; % beta
    freq_toi(2,:) = [58.2 68]; % gamma1
    freq_toi(3,:) = [80.8 87.1]; % gamma2
    freq_foi = zeros(length(freq_label),2); % 1: center frequency, 2: tapsmofrq
    freq_foi(1,:) = [21 8]; % beta
    freq_foi(2,:) = [40 10]; % gamma
    freq_foi(3,:) = [40 10]; % gamma
    freq_itr = 1:length(freq_label);
    for fi = freq_itr
        clear powcsd_* source_*
    %% 2. Time windows of interest
    cfg = [];
    cfg.toilim = freq_toi(fi,:);
    data_tw = ft_redefinetrial(cfg, data_clean);
    
    %% 3. calculating the cross spectral density matrix
    cfg = [];
    cfg.method    = 'mtmfft';
    cfg.taper     = 'dpss';
    cfg.output    = 'powandcsd';
    cfg.tapsmofrq = freq_foi(fi,2);
    cfg.foilim    = [freq_foi(fi,1) freq_foi(fi,1)]; % center frequency
    % for common filter over conditions and full duration
    powcsd_all      = ft_freqanalysis(cfg, data_tw);
    
    % for conditions
    cfg.trials       = I_HV;
    powcsd_HV        = ft_freqanalysis(cfg, data_tw);
    cfg.trials       = I_LV;
    powcsd_LV        = ft_freqanalysis(cfg, data_tw);
    
    %% 4. The forward model and lead field matrix
    %% 4.1. Head model
    
    % get headmodel
    load('standard_mri.mat');
    load('standard_bem.mat'); % colin27
    
    %% 4.2. compute lead field
    
    elec = ft_read_sens('standard_1020.elc');
    % compute the leadfields that can be used for the beamforming
    cfg            = [];
    cfg.elec       = elec;
    cfg.vol        = vol;
    cfg.reducerank      = 3; % default is 3 for EEG, 2 for MEG
    cfg.grid.resolution = 0.5;
    cfg.grid.unit       = 'cm';% same unit as above, i.e. in cm
    cfg.grid.tight      = 'yes';
    cfg.normalize = 'no';
    grid = ft_prepare_leadfield(cfg);
    
    % beamform common filter
    cfg              = [];
    cfg.method       = 'dics';
    cfg.elec         = elec;
    cfg.frequency    = freq_foi(fi,1); % center frequency
    cfg.grid         = grid;
    cfg.vol          = vol;
    cfg.channel = {'Fp1';'Fp2';'F7';'F3';'Fz';'F4';'F8';'FC5';'FC1';'FC2';'FC6';'T7';'C3';'Cz';'C4';'T8';'TP9';'CP5';'CP1';'CP2';'CP6';'TP10';'P7';'P3';'Pz';'P4';'P8';'PO9';'O1';'Oz';'O2';'PO10';'AF7';'AF3';'AF4';'AF8';'F5';'F1';'F2';'F6';'FT9';'FT7';'FC3';'FC4';'FT8';'FT10';'C5';'C1';'C2';'C6';'TP7';'CP3';'CPz';'CP4';'TP8';'P5';'P1';'P2';'P6';'PO7';'PO3';'POz';'PO4';'PO8'};
    cfg.dics.keepfilter   = 'yes';
    cfg.dics.lambda       = '15%';
    source_all = ft_sourceanalysis(cfg, powcsd_all);
    
    cfg.grid.filter = source_all.avg.filter;
    cfg.outputfile = fullfile(outputdir4, [subjectdata.subjectnr '_source_HV_' freq_label{fi}]);
    source_HV  = ft_sourceanalysis(cfg, powcsd_HV);
    cfg.outputfile = fullfile(outputdir4, [subjectdata.subjectnr '_source_LV_' freq_label{fi}]);
    source_LV = ft_sourceanalysis(cfg, powcsd_LV);
    
    
    
    if do_visualize
    sourceDiff = source_HV;
    sourceDiff.avg.pow = (source_HV.avg.pow - source_LV.avg.pow)./(source_HV.avg.pow + source_LV.avg.pow);
    cfg            = [];
    cfg.downsample = 2;
    cfg.parameter  = 'avg.pow';
    sourceDiffInt  = ft_sourceinterpolate(cfg, sourceDiff , mri);
    
    cfg = [];
    cfg.nonlinear     = 'no';
    sourceDiffIntNorm = ft_volumenormalise(cfg, sourceDiffInt);
    
    [atlas] = ft_read_atlas('D:\Hyun\fieldtrip-20170621\template\atlas\aal\ROI_MNI_V4.nii');
    cfg = [];
    cfg.method        = 'ortho'; % slice, ortho
    cfg.atlas         = atlas;
    cfg.interactive   = 'yes';
    cfg.funparameter  = 'pow';
    cfg.maskparameter = cfg.funparameter;
%     cfg.funcolormap    = 'jet';
    cfg.funcolorlim   = 'maxabs';
    cfg.opacitylim    = 'maxabs';
%     cfg.funcolorlim   = [-0.1 0.1];
%     cfg.opacitylim   = [0 0.1];
    cfg.opacitymap    = 'vdown';
    ft_sourceplot(cfg, sourceDiffIntNorm);
    
    cfg = [];
    cfg.method         = 'surface';
    cfg.funparameter   = 'pow';
    cfg.maskparameter  = cfg.funparameter;
%     cfg.funcolormap    = 'jet';
    cfg.funcolorlim   = 'maxabs';
%     cfg.opacitylim     = [0.0 0.1];
    cfg.opacitymap     = 'vdown';
    cfg.projmethod     = 'nearest';
    cfg.surffile       = 'surface_white_both.mat';
    cfg.surfdownsample = 10;
    ft_sourceplot(cfg, sourceDiffIntNorm);
    view ([90 0])
    end
    end
%     
   %% 2. arousal
%    
%     %%%%%%%%%%% arousal configurations
%     freq_label = {'beta','gamma'};
%     freq_toi = zeros(length(freq_label),2);
%     freq_toi(1,:) = [52.2 63.4]; % beta
%     freq_toi(2,:) = [101.4 115.5]; % gamma
%     freq_foi = zeros(length(freq_label),2); % 1: center frequency, 2: tapsmofrq
%     freq_foi(1,:) = [17 4]; % beta
%     freq_foi(2,:) = [40 10]; % gamma
%     freq_itr = 1:length(freq_label);
%     for fi = freq_itr
%         clear powcsd_* source_*
%     %% 2. Time windows of interest
%     cfg = [];
%     cfg.toilim = freq_toi(fi,:);
%     data_tw = ft_redefinetrial(cfg, data_clean);
%     
%     %% 3. calculating the cross spectral density matrix
%     cfg = [];
%     cfg.method    = 'mtmfft';
%     cfg.taper     = 'dpss';
%     cfg.output    = 'powandcsd';
%     cfg.tapsmofrq = freq_foi(fi,2);
%     cfg.foilim    = [freq_foi(fi,1) freq_foi(fi,1)]; % center frequency
%     % for common filter over conditions and full duration
%     powcsd_all      = ft_freqanalysis(cfg, data_tw);
%     
%     % for conditions
%     cfg.trials       = I_HA;
%     powcsd_HA        = ft_freqanalysis(cfg, data_tw);
%     cfg.trials       = I_LA;
%     powcsd_LA        = ft_freqanalysis(cfg, data_tw);
%     
%     %% 4. The forward model and lead field matrix
%     %% 4.1. Head model
%     
%     % get headmodel
%     load('standard_mri.mat');
%     load('standard_bem.mat'); % colin27
%     
%     %% 4.2. compute lead field
%     
%     elec = ft_read_sens('standard_1020.elc');
%     % compute the leadfields that can be used for the beamforming
%     cfg            = [];
%     cfg.elec       = elec;
%     cfg.vol        = vol;
%     cfg.reducerank      = 3; % default is 3 for EEG, 2 for MEG
%     cfg.grid.resolution = 0.5;
%     cfg.grid.unit       = 'cm';% same unit as above, i.e. in cm
%     cfg.grid.tight      = 'yes';
%     cfg.normalize = 'no';
%     grid = ft_prepare_leadfield(cfg);
%     
%     % beamform common filter
%     cfg              = [];
%     cfg.method       = 'dics';
%     cfg.elec         = elec;
%     cfg.frequency    = freq_foi(fi,1); % center frequency
%     cfg.grid         = grid;
%     cfg.vol          = vol;
%     cfg.channel = {'Fp1';'Fp2';'F7';'F3';'Fz';'F4';'F8';'FC5';'FC1';'FC2';'FC6';'T7';'C3';'Cz';'C4';'T8';'TP9';'CP5';'CP1';'CP2';'CP6';'TP10';'P7';'P3';'Pz';'P4';'P8';'PO9';'O1';'Oz';'O2';'PO10';'AF7';'AF3';'AF4';'AF8';'F5';'F1';'F2';'F6';'FT9';'FT7';'FC3';'FC4';'FT8';'FT10';'C5';'C1';'C2';'C6';'TP7';'CP3';'CPz';'CP4';'TP8';'P5';'P1';'P2';'P6';'PO7';'PO3';'POz';'PO4';'PO8'};
%     cfg.dics.keepfilter   = 'yes';
%     cfg.dics.lambda       = '15%';
%     source_all = ft_sourceanalysis(cfg, powcsd_all);
%     
%     cfg.grid.filter = source_all.avg.filter;
%     cfg.outputfile = fullfile(outputdir4, [subjectdata.subjectnr '_source_HA_' freq_label{fi}]);
%     source_HA  = ft_sourceanalysis(cfg, powcsd_HA);
%     cfg.outputfile = fullfile(outputdir4, [subjectdata.subjectnr '_source_LA_' freq_label{fi}]);
%     source_LA = ft_sourceanalysis(cfg, powcsd_LA);
%     
%     
%     
%     if do_visualize
%     sourceDiff = source_HA;
%     sourceDiff.avg.pow = (source_HA.avg.pow - source_LA.avg.pow)./(source_HA.avg.pow + source_LA.avg.pow);
%     cfg            = [];
%     cfg.downsample = 2;
%     cfg.parameter  = 'avg.pow';
%     sourceDiffInt  = ft_sourceinterpolate(cfg, sourceDiff , mri);
%     
%     cfg = [];
%     cfg.nonlinear     = 'no';
%     sourceDiffIntNorm = ft_volumenormalise(cfg, sourceDiffInt);
%     
%     [atlas] = ft_read_atlas('D:\Hyun\fieldtrip-20170621\template\atlas\aal\ROI_MNI_V4.nii');
%     cfg = [];
%     cfg.method        = 'ortho'; % slice, ortho
%     cfg.atlas         = atlas;
%     cfg.interactive   = 'yes';
%     cfg.funparameter  = 'pow';
%     cfg.maskparameter = cfg.funparameter;
% %     cfg.funcolormap    = 'jet';
%     cfg.funcolorlim   = 'maxabs';
%     cfg.opacitylim    = 'maxabs';
% %     cfg.funcolorlim   = [-0.1 0.1];
% %     cfg.opacitylim   = [0 0.1];
%     cfg.opacitymap    = 'vdown';
%     ft_sourceplot(cfg, sourceDiffIntNorm);
%     
%     cfg = [];
%     cfg.method         = 'surface';
%     cfg.funparameter   = 'pow';
%     cfg.maskparameter  = cfg.funparameter;
% %     cfg.funcolormap    = 'jet';
%     cfg.funcolorlim   = 'maxabs';
% %     cfg.opacitylim     = [0.0 0.1];
%     cfg.opacitymap     = 'vdown';
%     cfg.projmethod     = 'nearest';
%     cfg.surffile       = 'surface_white_both.mat';
%     cfg.surfdownsample = 10;
%     ft_sourceplot(cfg, sourceDiffIntNorm);
%     view ([90 0])
%     end
%     end
% %     
%    %% 3. liking
%    
%     %%%%%%%%%%% arousal configurations
%     freq_label = {'gamma'};
%     freq_toi = zeros(length(freq_label),2);
%     freq_toi(1,:) = [34.5 56.5]; % gamma
%     freq_foi = zeros(length(freq_label),2); % 1: center frequency, 2: tapsmofrq
%     freq_foi(1,:) = [40 10]; % beta
%     freq_itr = 1:length(freq_label);
%     for fi = freq_itr
%         clear powcsd_* source_*
%     %% 2. Time windows of interest
%     cfg = [];
%     cfg.toilim = freq_toi(fi,:);
%     data_tw = ft_redefinetrial(cfg, data_clean);
%     
%     %% 3. calculating the cross spectral density matrix
%     cfg = [];
%     cfg.method    = 'mtmfft';
%     cfg.taper     = 'dpss';
%     cfg.output    = 'powandcsd';
%     cfg.tapsmofrq = freq_foi(fi,2);
%     cfg.foilim    = [freq_foi(fi,1) freq_foi(fi,1)]; % center frequency
%     % for common filter over conditions and full duration
%     powcsd_all      = ft_freqanalysis(cfg, data_tw);
%     
%     % for conditions
%     cfg.trials       = I_like;
%     powcsd_like        = ft_freqanalysis(cfg, data_tw);
%     cfg.trials       = I_dislike;
%     powcsd_dislike        = ft_freqanalysis(cfg, data_tw);
%     
%     %% 4. The forward model and lead field matrix
%     %% 4.1. Head model
%     
%     % get headmodel
%     load('standard_mri.mat');
%     load('standard_bem.mat'); % colin27
%     
%     %% 4.2. compute lead field
%     
%     elec = ft_read_sens('standard_1020.elc');
%     % compute the leadfields that can be used for the beamforming
%     cfg            = [];
%     cfg.elec       = elec;
%     cfg.vol        = vol;
%     cfg.reducerank      = 3; % default is 3 for EEG, 2 for MEG
%     cfg.grid.resolution = 0.5;
%     cfg.grid.unit       = 'cm';% same unit as above, i.e. in cm
%     cfg.grid.tight      = 'yes';
%     cfg.normalize = 'no';
%     grid = ft_prepare_leadfield(cfg);
%     
%     % beamform common filter
%     cfg              = [];
%     cfg.method       = 'dics';
%     cfg.elec         = elec;
%     cfg.frequency    = freq_foi(fi,1); % center frequency
%     cfg.grid         = grid;
%     cfg.vol          = vol;
%     cfg.channel = {'Fp1';'Fp2';'F7';'F3';'Fz';'F4';'F8';'FC5';'FC1';'FC2';'FC6';'T7';'C3';'Cz';'C4';'T8';'TP9';'CP5';'CP1';'CP2';'CP6';'TP10';'P7';'P3';'Pz';'P4';'P8';'PO9';'O1';'Oz';'O2';'PO10';'AF7';'AF3';'AF4';'AF8';'F5';'F1';'F2';'F6';'FT9';'FT7';'FC3';'FC4';'FT8';'FT10';'C5';'C1';'C2';'C6';'TP7';'CP3';'CPz';'CP4';'TP8';'P5';'P1';'P2';'P6';'PO7';'PO3';'POz';'PO4';'PO8'};
%     cfg.dics.keepfilter   = 'yes';
%     cfg.dics.lambda       = '15%';
%     source_all = ft_sourceanalysis(cfg, powcsd_all);
%     
%     cfg.grid.filter = source_all.avg.filter;
%     cfg.outputfile = fullfile(outputdir, [subjectdata.subjectnr '_source_like_' freq_label{fi}]);
%     source_like  = ft_sourceanalysis(cfg, powcsd_like);
%     cfg.outputfile = fullfile(outputdir, [subjectdata.subjectnr '_source_dislike_' freq_label{fi}]);
%     source_dislike = ft_sourceanalysis(cfg, powcsd_dislike);
%     
%     
%     
%     if do_visualize
%     sourceDiff = source_like;
%     sourceDiff.avg.pow = (source_like.avg.pow - source_dislike.avg.pow)./(source_like.avg.pow + source_dislike.avg.pow);
%     cfg            = [];
%     cfg.downsample = 2;
%     cfg.parameter  = 'avg.pow';
%     sourceDiffInt  = ft_sourceinterpolate(cfg, sourceDiff , mri);
%     
%     cfg = [];
%     cfg.nonlinear     = 'no';
%     sourceDiffIntNorm = ft_volumenormalise(cfg, sourceDiffInt);
%     
%     [atlas] = ft_read_atlas('D:\Hyun\fieldtrip-20170621\template\atlas\aal\ROI_MNI_V4.nii');
%     cfg = [];
%     cfg.method        = 'ortho'; % slice, ortho
%     cfg.atlas         = atlas;
%     cfg.interactive   = 'yes';
%     cfg.funparameter  = 'pow';
%     cfg.maskparameter = cfg.funparameter;
% %     cfg.funcolormap    = 'jet';
%     cfg.funcolorlim   = 'maxabs';
%     cfg.opacitylim    = 'maxabs';
% %     cfg.funcolorlim   = [-0.1 0.1];
% %     cfg.opacitylim   = [0 0.1];
%     cfg.opacitymap    = 'vdown';
%     ft_sourceplot(cfg, sourceDiffIntNorm);
%     
%     cfg = [];
%     cfg.method         = 'surface';
%     cfg.funparameter   = 'pow';
%     cfg.maskparameter  = cfg.funparameter;
% %     cfg.funcolormap    = 'jet';
%     cfg.funcolorlim   = 'maxabs';
% %     cfg.opacitylim     = [0.0 0.1];
%     cfg.opacitymap     = 'vdown';
%     cfg.projmethod     = 'nearest';
%     cfg.surffile       = 'surface_white_both.mat';
%     cfg.surfdownsample = 10;
%     ft_sourceplot(cfg, sourceDiffIntNorm);
%     view ([90 0])
%     end
%     end
%     
end

if do_source_whole
    
    clearvars -except subjectdata Stim_i do_* single_subject_load pipe_mode i Subjectm outputdir*
    load([outputdir1 filesep subjectdata.subjectnr '_raw_clean.mat'],'data_clean');
    data_clean.trialinfo = table2array(data_clean.trialinfo);
%     %% compute the power spectrum
%     cfg              = [];
%     cfg.output       = 'pow';
%     cfg.method       = 'mtmfft';
%     cfg.taper        = 'dpss';
%     cfg.pad          = 'nextpow2';
%     cfg.tapsmofrq    = 1;
%     cfg.keeptrials   = 'no';
%     datapow          = ft_freqanalysis(cfg, data_clean);
    
    freq_bands = [5 7;8 12;14 20;20 30;30 50];
    freq_label = {'theta';'alpha';'low_beta';'high_beta';'gamma'};
    freq_foi = zeros(length(freq_bands),2); % 1: center frequency, 2: tapsmofrq
    freq_foi(1,:) = [6 1]; % beta
    freq_foi(2,:) = [10 2]; % gamma
    freq_foi(3,:) = [17 3]; % gamma
    freq_foi(4,:) = [25 5]; % gamma
    freq_foi(5,:) = [40 10]; % gamma
    
    source = cell(1,length(freq_bands));
    for fi = 1:length(freq_bands)
    %% 3. calculating the cross spectral density matrix
    cfg = [];
    cfg.method    = 'mtmfft';
    cfg.output    = 'fourier';
    cfg.tapsmofrq = freq_foi(fi,2);
    cfg.foi       = freq_foi(fi,1); % center frequency
    cfg.keeptrials = 'yes';
    % for common filter over conditions and full duration
    freq      = ft_freqanalysis(cfg, data_clean);
    %% The forward model and lead field matrix
    %% 1. Head model
    
    % get headmodel
    load('standard_mri.mat');
    load('standard_bem.mat'); % colin27
    
    %% 2. compute lead field
    
    elec = ft_read_sens('standard_1020.elc');
    % compute the leadfields that can be used for the beamforming
    cfg            = [];
    cfg.elec       = elec;
    cfg.vol        = vol;
    cfg.reducerank      = 3; % default is 3 for EEG, 2 for MEG
    cfg.grid.resolution = 0.5;
    cfg.grid.unit       = 'cm';% same unit as above, i.e. in cm
    cfg.grid.tight      = 'yes';
    cfg.normalize = 'no';
    grid = ft_prepare_leadfield(cfg);
    
    % beamform common filter
    cfg              = [];
    cfg.method       = 'pcc';
    cfg.elec         = elec;
    cfg.frequency         = freq.freq;
    cfg.grid         = grid;
    cfg.vol          = vol;
%     cfg.headmodel    = vol;
    cfg.channel = {'Fp1';'Fp2';'F7';'F3';'Fz';'F4';'F8';'FC5';'FC1';'FC2';'FC6';'T7';'C3';'Cz';'C4';'T8';'TP9';'CP5';'CP1';'CP2';'CP6';'TP10';'P7';'P3';'Pz';'P4';'P8';'PO9';'O1';'Oz';'O2';'PO10';'AF7';'AF3';'AF4';'AF8';'F5';'F1';'F2';'F6';'FT9';'FT7';'FC3';'FC4';'FT8';'FT10';'C5';'C1';'C2';'C6';'TP7';'CP3';'CPz';'CP4';'TP8';'P5';'P1';'P2';'P6';'PO7';'PO3';'POz';'PO4';'PO8'};
    cfg.pcc.lambda        = '10%';
    cfg.pcc.projectnoise  = 'yes';
    cfg.pcc.fixedori      = 'yes';
    cfg.keeptrials        = 'yes';
    cfg.pcc.keepfilter   = 'yes';
    source = ft_sourceanalysis(cfg, freq);
    
    parc_conn = cell(1,length(freq.trialinfo));
    
    for ti = 1:length(freq.trialinfo)
        clear source_conn
        cfg = [];
        cfg.trials = ti;
        freq_single = ft_selectdata(cfg, freq);
        
        % use the precomputed filters
        cfg                   = [];
        cfg.method            = 'pcc';
        cfg.elec         = elec;
        cfg.frequency         = freq.freq;
        cfg.grid              = grid;
        cfg.vol          = vol;
        cfg.grid.filter       = source.avg.filter;
        cfg.channel = {'Fp1';'Fp2';'F7';'F3';'Fz';'F4';'F8';'FC5';'FC1';'FC2';'FC6';'T7';'C3';'Cz';'C4';'T8';'TP9';'CP5';'CP1';'CP2';'CP6';'TP10';'P7';'P3';'Pz';'P4';'P8';'PO9';'O1';'Oz';'O2';'PO10';'AF7';'AF3';'AF4';'AF8';'F5';'F1';'F2';'F6';'FT9';'FT7';'FC3';'FC4';'FT8';'FT10';'C5';'C1';'C2';'C6';'TP7';'CP3';'CPz';'CP4';'TP8';'P5';'P1';'P2';'P6';'PO7';'PO3';'POz';'PO4';'PO8'};
        cfg.pcc.lambda        = '10%';
        cfg.pcc.projectnoise  = 'yes';
        source_single  = ft_sourcedescriptives([], ft_sourceanalysis(cfg, freq_single));
        
        %% compute connectivity
        cfg         = [];
        cfg.method  ='coh';
        cfg.complex = 'absimag';
        source_conn = ft_connectivityanalysis(cfg, source_single);
        %     figure;imagesc(source_conn.cohspctrm);
        
        aal =ft_read_atlas ('atlas\aal\ROI_MNI_V5.nii');
        %     harvard =ft_read_atlas ('atlas\HarvardOxford-cort-maxprob-thr0-1mm.nii');
        % ref: http://xujiahua.cn/source-or-connecitvity-parcellation-in-filedtrip/
        % and call ft_sourceinterpolate:
        cfg = [];
        cfg.interpmethod = 'nearest';
        cfg.parameter = 'tissue';
        sourcemodel2 = ft_sourceinterpolate(cfg, aal, grid);
        sourcemodel2.pos = source_conn.pos; % otherwise the parcellation won't work
        
        cfg = [];
        cfg.parcellation = 'tissue';
        cfg.parameter    = 'cohspctrm';
        parc_conn{ti} = ft_sourceparcellate(cfg, source_conn, sourcemodel2);
    end
    
    save(['Analysis_EEG' filesep 'Result_emotion_source_pcc' filesep subjectdata.subjectnr '_source_conn_' freq_label{fi} '.mat'],'parc_conn','grid','elec','freq');
    
    end
    
    
    
    
%     cfg           = [];
% cfg.method    = 'degrees';
% cfg.parameter = 'cohspctrm';
% cfg.threshold = .1;
% network_full = ft_networkanalysis(cfg,source_conn);
% network_parc = ft_networkanalysis(cfg,parc_conn);
% 
% %% visualize
% cfg               = [];
% cfg.method        = 'surface';
% cfg.funparameter  = 'degrees';
% cfg.funcolormap   = 'jet';
% ft_sourceplot(cfg, network_full);
% view([-150 30]);
% 
% ft_sourceplot(cfg, network_parc);
% view([-150 30]);

    
%     atlas_parc = ft_datatype_parcellation(atlas);
%     atlas_parc.pos = source_conn.pos; % otherwise the parcellation won't work
%     cfg = [];
%     cfg.parcellation = 'parcellation';
%     cfg.parameter    = 'cohspctrm';
%     parc_conn = ft_sourceparcellate(cfg, source_conn, atlas);
% 
%     load atlas_MMP1.0_4k.mat;
%     atlas.pos = source_conn.pos; % otherwise the parcellation won't work
%     
%     cfg = [];
%     cfg.parcellation = 'parcellation';
%     cfg.parameter    = 'cohspctrm';
%     parc_conn = ft_sourceparcellate(cfg, source_conn, atlas);
% 
%     figure;imagesc(parc_conn.cohspctrm);
%     
% cfg = [];
% cfg.method        = 'surface';
% cfg.funparameter  = 'pow';
% % cfg.maskparameter = 'mask';
% % cfg.funcolorlim   = [-.3 .3];
% cfg.funcolormap   = 'jet';
% cfg.colorbar      = 'no';
% ft_sourceplot(cfg, source);
% view([-90 30]);
% light('style','infinite','position',[0 -200 200]);

    
end

if do_svm % For ML
    % feature extraction
    % ERP feature
    load([outputdir1 filesep subjectdata.subjectnr '_raw_clean.mat']);
    % baseline correction  & filtering
    cfg = [];
    cfg.lpfilter        = 'yes';
    cfg.lpfreq          = 30;
    cfg.demean          = 'yes';
    cfg.baselinewindow  = [-0.2 0];
    data_cue_locked       = ft_preprocessing(cfg,data_clean); % for Cue condition
    
    ERP_f = data_cue_locked;
    wsample = 20; % 50 ms.
    for ti = 1:length(ERP_f.trial)
        ERP_f.trial{ti} = [];
        for ci = 1:length(data_cue_locked.time{1})/wsample
            ERP_f.trial{ti}(:,ci) = nanmean(data_cue_locked.trial{ti}(:,wsample*(ci-1)+1:wsample*ci),2);
        end
        ERP_f.trial{ti} = reshape(ERP_f.trial{ti},1,size(ERP_f.trial{ti},1)*size(ERP_f.trial{ti},2));
    end
    label_ch = [];
    label_ch = repmat(ERP_f.label',1,length(data_cue_locked.time{1})/wsample);
    label_t = [];
    for tmi = 1:length(data_cue_locked.time{1})/wsample
        label_t(1,length(ERP_f.label)*(tmi-1)+1:length(ERP_f.label)*tmi) = tmi*ones(1,length(ERP_f.label));
    end
    ERP_f.trialinfo = [];
    for fi = 1:length(ERP_f.trial{1})
        ERP_f.trialinfo{fi} = ['ERP_' cell2mat(label_ch(fi)) '_' num2str((label_t(fi)-1)*(1000/wsample)) '_' num2str(label_t(fi)*(1000/wsample)) 'ms'];
    end
    
    % ERSP feature
    freq.type = 'hann'; % analysis method
    switch freq.type
        case 'wavelet' % need to adjust
        case 'hann'
            cfg              = [];
            cfg.output       = 'pow';
            cfg.method       = 'mtmconvol';
            cfg.pad = 'nextpow2';
            cfg.taper        = 'hanning';
            cfg.foi          = 1:1:50;                         % analysis 2 to 30 Hz in steps of 2 Hz
            cfg.t_ftimwin    = ones(length(cfg.foi),1).*0.1;   % length of time window (ex. 0.5 sec)
            cfg.keeptrials = 'yes';
            cfg.toi          = -0.4:0.05:2.2;                  % time window "slides" from -0.5 to 1.5 sec in steps of 0.05 sec (50 ms)
            freq = ft_freqanalysis(cfg, data_clean);
    end
    
    %%% log power conversion for more sensitivity
    %   cfg = [];
    %   cfg.parameter = 'powspctrm';
    %   cfg.operation = 'log10';
    %   freq_log10 = ft_math(cfg,freq);
    %%% Baseline correction
    cfg           = [];
    cfg.baseline = [-0.3 -0.1];
    cfg.baselinetype = 'relchange';
    freq_bc = ft_freqbaseline(cfg,freq);
        
%     cfg = [];
%     cfg.parameter = 'powspctrm';
%     cfg.channel = 'PZ';
%     figure;ft_singleplotTFR(cfg,freq_bc_t);

ERSP_f = freq_bc;
freq_bands = {'theta','alpha','beta','gamma'};
foi_group = [4 8;9 13;13 30;30 50];
label_ch = [];
label_ch = repmat(ERSP_f.label',1,length(ERSP_f.time));
label_t = [];
label_t = repmat(ERSP_f.time,60,1);
label_t = reshape(label_t,1,size(label_t,1)*size(label_t,2));
label_f = repmat(freq_bands, size(label_t,1)*size(label_t,2),1);
label_f = reshape(label_f,1,size(label_f,1)*size(label_f,2));
for ti = 1:size(freq_bc.powspctrm,1)
    F = [];
for fi = 1:4
    foi = foi_group(fi,:);
    freq_i = (foi(1)<freq_bc.freq & freq_bc.freq<foi(end));
    avg_freq_bc = squeeze(nanmean(freq_bc.powspctrm(ti,:,freq_i,:),3));
    F_avg = reshape(avg_freq_bc,1,size(avg_freq_bc,1)*size(avg_freq_bc,2));
    F = [F reshape(avg_freq_bc,1,size(avg_freq_bc,1)*size(avg_freq_bc,2))];
end
    ERSP_f.trial{ti} = F;
end
L = [];
for fti = 1:length(F_avg)
    L{fti} = [cell2mat(label_ch(fti)) '_' num2str(label_t(fti)) 's'];
end
L = repmat(L,1,4);
ERSP_f.trialinfo = [];
for fti = 1:length(label_f)
    ERSP_f.trialinfo{fti} = [label_f{fti} '_' L{fti}];
end

% feature combination
Total_F = ERP_f;
for ti = 1:length(ERP_f.trial)
    Total_F.trial{ti} = [ERP_f.trial{ti} ERSP_f.trial{ti}];
end
Total_F.trialinfo = [ERP_f.trialinfo ERSP_f.trialinfo];

save([outputdir5 filesep subjectdata.subjectnr '_features.mat'],'Total_F');

end