% Takes in manually labeled points, undistorts the points, and triangulates
% to 3D using camera calibration info

calibfile = '/home/twd/Dropbox/mocapdata_for_tim/CameraCalib_rd16_20190618/20190617_calibration_pup/worldcoordinates_lframe.mat';
CROP_HEIGHT = 0;%These offsets are critical! TODO: port this to python and
% make sure this is read in from config file
CROP_WIDTH = 20;%These offsets are critical! TODO: port this to python and
% make sure this is read in from config file
load(calibfile);
cams = [2, 3, 1]; %This ordering is critical! TODO: port this to python and
% make sure this is read in from config file
camnames = {'CameraLmouse', 'CameraRmouse', 'CameraSmouse'};
data = {};
data_undistorted = {};
data_raw = {};
for i = 1:numel(cams)
    load([camnames{i} '_manlabels_worker0.mat']);
    data{i} = data2d;
    data_reshape = reshape(data2d,[size(data2d,1)*size(data2d,2),2]);
    data_reshape(:,2) = data_reshape(:,2) + CROP_HEIGHT;
    data_reshape(:,1) = data_reshape(:,1) + CROP_WIDTH;
    data_raw{i} = reshape(data_reshape,[size(data2d,1),size(data2d,2),2]);
    inds = find(isnan(data_reshape(:,1)));
    data_reshape(inds,:) = 0;
    data_undistorted{i} = undistortPoints(data_reshape,params_individual{cams(i)});
    data_undistorted{i}(inds,:) = NaN;
    data_undistorted{i} = reshape(data_undistorted{i},[size(data2d,1),size(data2d,2),2]);
end

%%
campairs = {};
cnt = 1;
for i = 1:numel(cams)
    j = i+1;
    while j <= numel(cams)
        cam_i = cams(i);
        cam_j = cams(j);
        cammat_i = cameraMatrix(params_individual{cam_i},rotationMatrix{cam_i},translationVector{cam_i});
        cammat_j = cameraMatrix(params_individual{cam_j},rotationMatrix{cam_j},translationVector{cam_j});
        
        %Now get respective data, reshape, replace nans, triangulate, shape
        %back
        data_i = data_undistorted{i};
        data_i_reshape = reshape(data_i,[size(data_i,1)*size(data_i,2),2]);
        
        data_j = data_undistorted{j};
        data_j_reshape = reshape(data_j,[size(data_j,1)*size(data_j,2),2]);
        
        inds = find(isnan(data_i_reshape(:,1)) | isnan(data_j_reshape(:,1)));
        display(size(inds));
        data_i_reshape(inds,:) = 0;
        data_j_reshape(inds,:) = 0; 
        
        world = triangulate(data_i_reshape,data_j_reshape,cammat_i,cammat_j);
        world(inds,:) = nan;
        campairs{cnt} = reshape(world,[size(data_i,1),size(data_i,2),3]);
        cnt = cnt +1;
        j = j+1;
    end
end

%% Save data structures with 2d & 3d data, along with sampleIDs, for training
% nets.
basedir = './';
data_3d = [];
for j=1:numel(campairs)
    data_3d = cat(4,data_3d,campairs{j});
end
data_3d = nanmedian(data_3d,4);

%% Save data structures with 2d & 3d data, along with sampleIDs, for training
% nets. WE CANNOT GET DATA_FRAME NAIVELY BY DIVIDING BY FRAME PERIOD
% BECAUSE OF ROUNDING ERRORS and CAMERA blips. Must load in the frames
% from matched frames. If not, significant offsets can result when training

matched_frames_dir = '/home/twd/Dropbox/mocapdata_for_tim/CameraCalib_rd16_20190618/RecordingP7Pup_three/';
matched_frames_sub = '_MatchedFrames.mat';
basedir = './';
data_3d = [];
for j=1:numel(campairs)
    data_3d = cat(4,data_3d,campairs{j});
end
data_3d = nanmedian(data_3d,4);
data_sampleID = sampleID;
wpts = reshape(data_3d,[size(data_3d,1)*size(data_3d,2),size(data_3d,3)]);
data_3d = reshape(permute(data_3d,[1,3,2]),[size(data_3d,1),size(data_3d,2)*size(data_3d,3)]);

for camerause = 1:3
    % For each cameras, project to 2D and save the results
    d = load([matched_frames_dir camnames{camerause} matched_frames_sub]);
    data_frame = [];
    for i = 1:numel(data_sampleID)
        data_frame(i) = d.data_frame(d.data_sampleID==data_sampleID(i));
    end
    ipts = worldToImage(params_individual{cams(camerause)},rotationMatrix{cams(camerause)}, ...
        translationVector{cams(camerause)}, wpts,'ApplyDistortion',true);
    data_2d = reshape(ipts,[size(campairs{1},1),size(campairs{1},2),2]);
    data_2d = reshape(permute(data_2d,[1,3,2]),[size(data_2d,1),size(data_2d,2)*size(data_2d,3)]);
    save([basedir camnames{camerause} '_data_applyDistort.mat'],'data_frame','data_2d','data_3d','data_sampleID');
    
end