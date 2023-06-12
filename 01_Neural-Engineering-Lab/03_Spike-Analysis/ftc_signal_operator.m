function data = ftc_signal_operator(cfg, data)
% noise level estimation
% input data: bandpass-filtered signal

if ~isfield(cfg, 'method')
    cfg.method = 'square';
end

switch cfg.method
    case 'square'
        for ti = 1:length(data.trial)
            timeseries = data.trial{1,ti};
            data.trial{1,ti} = sign(timeseries).*abs(timeseries).^2;
        end
end