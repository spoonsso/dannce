%quick script to undistort an allCOMs file using matlab's faster
% undistort function. Undistorted data are re-saved for python

function undistort_allCOMS(comfile,calibfile,camorder)
load(comfile,'allCOMs');

%calibfile = '/home/twd/Dropbox/mocapdata_for_tim/CameraCalib_rd12_20190508_mice/20190508/20190405_calibration_rat/worldcoordinates_lframe.mat';
load(calibfile,'params_individual');
%cam order was E, S, U2, R, L ,U, i.e. for cam numbers :3,5,6,1,2,4
%camorder = [2, 3, 1]
allCOMs_u = zeros(size(allCOMs,1),size(allCOMs,2),size(allCOMs,3));
chunk = 5000;
for i=1:numel(camorder)
        camn = camorder(i);
    thiscam = params_individual{camn};
    display(i)
    for j = 1:chunk:size(allCOMs,2)
    display(j)
    endy = min([j+chunk,size(allCOMs,2)]);
    allCOMs_u(i,j:endy,:) = ...
        undistortPoints(squeeze(allCOMs(i,j:endy,:)),thiscam);
    end
end

save allCOMs_undistorted allCOMs_u
