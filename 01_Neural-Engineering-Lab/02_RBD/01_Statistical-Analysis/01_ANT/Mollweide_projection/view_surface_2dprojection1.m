function [hFig, iDS, iFig] = view_surface_2dprojection_old(ResultsFile)
% VIEW_SURFACE_SPHERE: Display the registration sphere/square for a surface.
%
% USAGE:  [hFig, iDS, iFig] = view_surface(SurfaceFile)
%         [hFig, iDS, iFig] = view_surface(ResultsFile)

% @=============================================================================
% This function is part of the Brainstorm software:
% https://neuroimage.usc.edu/brainstorm
% 
% Copyright (c)2000-2019 University of Southern California & McGill University
% This software is distributed under the terms of the GNU General Public License
% as published by the Free Software Foundation. Further details on the GPLv3
% license can be found at http://www.gnu.org/copyleft/gpl.html.
% 
% FOR RESEARCH PURPOSES ONLY. THE SOFTWARE IS PROVIDED "AS IS," AND THE
% UNIVERSITY OF SOUTHERN CALIFORNIA AND ITS COLLABORATORS DO NOT MAKE ANY
% WARRANTY, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF
% MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, NOR DO THEY ASSUME ANY
% LIABILITY OR RESPONSIBILITY FOR THE USE OF THIS SOFTWARE.
%
% For more information type "brainstorm license" at command prompt.
% =============================================================================@
%
% Authors: Francois Tadel, 2013-2015
% Initialize returned variables
global GlobalData;
hFig = [];
iDS  = [];
iFig = [];

% ===== LOAD DATA =====
% Display progress bar
isProgress = ~bst_progress('isVisible');
if isProgress
    bst_progress('start', 'View surface', 'Loading surface file...');
end

% Get file type
fileType = file_gettype(ResultsFile);

% If it is a results file
if ismember(fileType, {'results','link','ptimefreq','timefreq'})
    ResultsMat = in_bst_results(ResultsFile, 0, 'SurfaceFile');
    SurfaceFile = ResultsMat.SurfaceFile;
else
    ResultsFile = [];
end

% ===== LOAD REGISTRATION =====
% Load vertices
TessMat = in_tess_bst(SurfaceFile);
if isfield(TessMat, 'Reg') && isfield(TessMat.Reg, 'Sphere') && isfield(TessMat.Reg.Sphere, 'Vertices') && ~isempty(TessMat.Reg.Sphere.Vertices)
    sphVertices = double(TessMat.Reg.Sphere.Vertices);
    lrOffset = 0.12;
else
    bst_error('There is no registered sphere available for this surface.', 'View registered sphere/square', 0);
    return;
end

% % Get subject MRI file
% sSubject = bst_get('SurfaceFile', SurfaceFile);
% sMri = in_mri_bst(sSubject.Anatomy(1).FileName);
% % Convert: FreeSurfer RAS coord => Voxel
% mriSize = size(sMri.Cube) / 2;
% sphVertices = bst_bsxfun(@plus, sphVertices .* 1000, mriSize);
% % Convert: Voxel => SCS
% sphVertices = cs_convert(sMri, 'voxel', 'scs', sphVertices);

% Detect the two hemispheres
[ir, il, isConnected] = tess_hemisplit(TessMat);
% If there is a Structures atlas with left and right hemispheres: split in two spheres
if isempty(ir) && isempty(il) && isConnected
    error('hello')
end

% Get the latidude and longitude coordinates of the vertices of each
% hemisphere
[il_lon, il_lat, il_radius] = cart2sph(sphVertices(il, 1), sphVertices(il, 2), sphVertices(il, 3));
[ir_lon, ir_lat, ir_radius] = cart2sph(sphVertices(ir, 1), sphVertices(ir, 2), sphVertices(ir, 3));

% Get the 2D projection coordinates using mollweide projection
[il_x, il_y] = mollweide_proj(il_lat, il_lon, mean(il_radius(:)));
[ir_x, ir_y] = mollweide_proj(ir_lat, ir_lon, mean(ir_radius(:)));

% Replace the spherical coordinates with the flattened coordinates
twoDVertices = sphVertices;
twoDVertices(il, 1) = il_x;
twoDVertices(il, 2) = il_y;
twoDVertices(ir, 1) = ir_x;
twoDVertices(ir, 2) = ir_y;
twoDVertices(ir, 3) = 0;
twoDVertices(il, 3) = 0;

ProtocolInfo = bst_get('ProtocolInfo');
outmat = load(bst_fullfile(ProtocolInfo.SUBJECTS, SurfaceFile));
outmat.comment = '2D projection_cortex';
outmat.Vertices = twoDVertices;
FileName = bst_fullfile(ProtocolInfo.SUBJECTS, fileparts(SurfaceFile), 'tess_2D_cortex.mat');
if ~isfile(FileName)
    bst_save(FileName, outmat);
    [sSubject, iSubject, iFirstSurf] = bst_get('SurfaceFile', SurfaceFile);
    Filename = file_short(FileName);
    db_add_surface(iSubject, Filename, outmat.comment, sSubject.Surface(iFirstSurf).SurfaceType);
end

% % ===== DISPLAY SURFACE =====
% Display surface only
if isempty(ResultsFile)
    TessMat.Vertices = twoDVertices;
    [hFig, iDS, iFig] = view_surface_matrix(TessMat);
% Display surface + results
else
    % Open cortex with results
    [hFig, iDS, iFig] = view_surface_data(FileName, ResultsFile, [], 'NewFigure');
end

% Show figure
set(hFig, 'Visible', 'on');
if isProgress
    bst_progress('stop');
end



