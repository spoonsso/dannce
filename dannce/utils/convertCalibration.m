%% this simple script loads in a world coordinates calibration file
% and saves into structures used by DANNCE.
%
% Use: navigate to folder containing a calibration file in Jesse format, then run this script. 
load('worldcoordinates_lframe.mat')

for camerause = 1:6
    RDistort = params_individual{camerause}.RadialDistortion;
    TDistort = params_individual{camerause}.TangentialDistortion;
    r = rotationMatrix{camerause};
    t = translationVector{camerause};
    K = params_individual{camerause}.IntrinsicMatrix;
    save(['hires_cam' num2str(camerause) '_params.mat'],'K','r','t','RDistort','TDistort')
end
