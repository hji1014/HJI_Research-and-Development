function Data = mat2fieldtrip(cfg)

% conversion .mat file to filedtrip dataformat
% for simulated data by Quiroga
% [Data] = mat2fieldtrip(cfg, data)
% cfg must need 'cfg.path = data path'
% cfg must need 'cfg.label = channel information'
% Data must have at least 1 channel (unit[?])
% Data : channel X time

if nargin < 1
	disp('Not enough input arguments');
	return;
end

fnames = dir(cfg.path);

if numel(fnames) < 3
    error('no file exists');
end

fdataname = cell(0,1);
for fi = 3:length(fnames)
    if (strfind(fnames(fi).name, cfg.trialtype)) & (isempty(strfind(fnames(fi).name, 'short'))) & (isempty(strfind(fnames(fi).name, 'times')))
        fdataname = cat(1, fdataname, fnames(fi).name);
    end
end

cfg.ntrials = length(fdataname);
number_of_trials = cfg.ntrials;

Data = struct;
Data.fsample = 24000;
Data.label = cfg.label;

for tri = 1:number_of_trials
    load([cfg.path filesep fdataname{tri,1}]);
    Data.trial{tri} = data;
    align_time = 0:1/Data.fsample:((size(data,2)/Data.fsample)-1/Data.fsample);
    Data.time{tri} = align_time;
    nlevel = sscanf(fdataname{tri,1}, ['C_' cfg.trialtype '_noise%d.mat']);
    switch nlevel
        case {1, 2, 3, 4}
            nlevel = nlevel*10;
    end
    ntype = num2str(nlevel/100);
    Data.trialinfo{tri} = [cfg.trialtype '_' ntype];
    Data.cfg.overlap_data{tri} = OVERLAP_DATA;
    Data.cfg.spike_class{tri} = spike_class;
    Data.cfg.spike_times{tri} = spike_times;
end

