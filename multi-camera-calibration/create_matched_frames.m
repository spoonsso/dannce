%% Use this to create a Jesse-like matched_frames precursor
clear; clc

baseFolder = 'D:\\20191030\mouse11';
cd(baseFolder)

camnames{1} = 'Camera1';
camnames{2} = 'Camera2';
camnames{3} = 'Camera3';
camnames{4} = 'Camera4';
camnames{5} = 'Camera5';
camnames{6} = 'Camera6';

% num_frames is the total number of frames, across all videos (for a single camera), for a given
% animal
num_frames = 360000;
fr = 100;
fp = 1000/fr;
l = num_frames*fp;

mframes = 1:l;
mframes = floor(mframes/fp) + 1; %for matlab 1-indexing

matched_frames_aligned = {};
for i=1:numel(camnames)
    matched_frames_aligned{i} = mframes;
end
save matchedframes camnames matched_frames_aligned

% load
mdir = [baseFolder filesep 'data' filesep];
num_markers = 22; %hard-coded for mouse
frame_period = 10; %in ms
for camerause = 1:numel(camnames)
    clear data_2d data_3d
    frame_inds = 1:frame_period:length(matched_frames_aligned{camerause});
    frame_inds = round(frame_inds);    
    data_sampleID = nan(length(frame_inds),1);
    data_frame = nan(length(frame_inds),1);
    data_2d = zeros(length(frame_inds),2*num_markers);
    data_3d = zeros(length(frame_inds),3*num_markers);
    
    cnt = 1;
    for frame_to_plot = frame_inds
        thisinds = frame_to_plot;    
        data_sampleID(cnt) = frame_to_plot;
        data_frame(cnt) = matched_frames_aligned{camerause}(frame_to_plot)-1;    
        cnt = cnt +1;
    end
    data_frame(data_frame<0) = 0;
    save([mdir camnames{camerause} '_MatchedFrames'],'data_frame','data_2d','data_3d','data_sampleID');
end