function label = func_identify_brain_region(cfg, sourcevalues)

% sourcevalues: values of source activity extracted from anatomy
% cfg.atlas: brain atlas struct (e.g. 'Mindboogle', 'Desikan_Killiany')
% cfg.mni: mni coordinates (should equal to the length of sourcevalues)

% extract indices
if numel(find(isnan(sourcevalues)==1))
    warning('nan values set to 0');
    sourcevalues(isnan(sourcevalues)) = 0;
end

source_rank = sort(sourcevalues(:), 'descend');
thd = source_rank(cfg.max_len);

[idx, row] = find(sourcevalues>thd); 

label = cell(length(idx), 3);
% search
for id = 1:length(idx)
    for ai = 1:length(cfg.atlas)
        if ismember(idx(id),cfg.atlas(ai).Vertices)
            label{id,1} = cfg.atlas(ai).Label; % label name
            label{id,2} = idx(id); % vertex index
            label{id,3} = cfg.mni(idx(id), :); % mni coordinate
            label{id,4} = sourcevalues(idx(id)); % value
        end
    end
end

