%quick script to undistort an allCOMs file using matlab's faster
% undistort function. Undistorted data are re-saved for python

function undistort_allCOMS(comfile,calibfiles)
load(comfile,'allCOMs');

allCOMs_u = zeros(size(allCOMs,1),size(allCOMs,2),size(allCOMs,3));
chunk = 5000;
for i=1:numel(calibfiles)
    load(calibfiles{i});
    params = cameraParameters('IntrinsicMatrix',K,...
        'RadialDistortion',RDistort,...
        'TangentialDistortion',TDistort);
    display(i)
    for j = 1:chunk:size(allCOMs,2)
    display(j)
    endy = min([j+chunk,size(allCOMs,2)]);
    allCOMs_u(i,j:endy,:) = ...
        undistortPoints(squeeze(allCOMs(i,j:endy,:)),params);
    end
end

save allCOMs_undistorted allCOMs_u
