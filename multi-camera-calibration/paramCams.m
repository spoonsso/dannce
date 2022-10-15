clear

numCams = 6;
baseFolder = 'D:\\20191122\mouse';

load([baseFolder filesep 'calibration\camera_params.mat'])
load([baseFolder filesep 'calibration\intrinsic\cam_intrinsics.mat'])
% extrinsics
% r = rotation matrix
% t = translation matrix

% intrinsics
% K = intrinsic matrix
% RDistort = RadialDistortion
% TDistort = TangentialDistortion

for i = 1:numCams
    outputFolder = [baseFolder filesep 'calibration'];
    r = rotationMatrix{i};
    t = translationVector{i};
    K = params_individual{i}.IntrinsicMatrix;
    RDistort = params_individual{i}.RadialDistortion;
    TDistort = params_individual{i}.TangentialDistortion;
    save([outputFolder filesep 'kyle_cam' num2str(i) '_params'],'r','t','K','RDistort','TDistort')
end





