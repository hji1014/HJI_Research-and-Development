% Mollweide projection example code for source result using shared kernel

ABS_PATH = 'brainstorm_db_path';
ni = 1; % subject number
ri = 1; % resultdata number

Subjectm = sprintf('Subject%02d',ni);
disp(['now processing: ' Subjectm]);
[sStudy, iStudy] = bst_get('StudyWithCondition', [Subjectm filesep 'condition_name']);
sSubject = bst_get('Subject', Subjectm);
SurfaceFile = [ABS_PATH filesep 'anat' filesep sSubject.Surface(sSubject.iCortex).FileName];
FileNames = strsplit(sStudy.Result(ri).FileName,'|');
SourceName = [ABS_PATH filesep 'data' filesep FileNames{2,1}];
DataName = [ABS_PATH filesep 'data' filesep FileNames{3,1}];
size_x = 60; size_y = 120; % size of mollweide projection 2D image
[lH_image, rH_image] = flatten_source(SourceName, SurfaceFile, size_x, size_y, DataName);