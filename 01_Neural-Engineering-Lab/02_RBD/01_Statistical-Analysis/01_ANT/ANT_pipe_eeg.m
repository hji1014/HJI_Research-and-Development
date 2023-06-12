clear all;clc;
cd G:\2019_과제\ANT\ft_pipeline
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Subject load
do_visualize      = false; %
do_baseline       = true; % baseline correction for TFR
do_baseline_time  = false; % baseline correction for ER , default : false
single_subject_load = false; % single or all analysis
pipe_mode = 'none';
if single_subject_load
    subj = 1; % choose single subject
    Subjects = subj;
else
    subj = 47;
    Subjects = 1:subj;
end    
for i = Subjects
    Subjectm = sprintf('subject%02d', i);
    eval(Subjectm);
    if ~subjectdata.eeg.stat
        disp(['subject' num2str(subj) ' is excluded.']);
        continue;
    else
        eval('eeg_analyze_single_subject')
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Grand analysis
do_visualize      = true; %
eval([Stim_i '_ppg_analyze_group'])

%% look at the analysis history
% cfg           = [];
% cfg.filename  = [Stim_i '_ppg_diff_rmb_vs_frg.html'];
% ft_analysispipeline(cfg, diff_rmb_vs_frg);
% cfg           = [];
% cfg.filename  = [Stim_i '_ppg_TFR_stat_rmb_vs_frg.html'];
% ft_analysispipeline(cfg, TFR_stat_rmb_vs_frg);

