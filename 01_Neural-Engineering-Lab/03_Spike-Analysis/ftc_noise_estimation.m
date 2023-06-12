function data = ftc_noise_estimation(cfg, data)
% noise level estimation
% input data: bandpass-filtered signal
% cfg.method: 'std'(Donoho & Johnstone, 1994), 'median'(Quiroga, 2004),
% method 참고: doi 10.1162/089976604774201631

if ~isfield(cfg, 'method')
    cfg.method = 'median';
end
if isfield(cfg, 'threshold')
    data.threshold = cell(1,length(data.trial));
else
    cfg.threshold = 'no';
end

data.noiseinfo = cell(1,length(data.trial));
for ti = 1:length(data.trial)
    timeseries = data.trial{1,ti};
    bkgnoise = abs(timeseries)/0.6745;
    switch cfg.method
        case 'std'
            estimate = std(bkgnoise);
        case 'median'
            estimate = median(bkgnoise);
    end
    data.noiseinfo{1,ti} = estimate;
    if strcmp(cfg.threshold, 'auto')
        data.threshold{1,ti} = 4*estimate;
    end
end
