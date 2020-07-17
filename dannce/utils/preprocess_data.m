%% processes matched_frames_aligned to provide a data structure used for model training
% Note that the model data files require marker label data, which we will
% just set to NaN here.
clear all; close all;
% paths to matched frames
dirname = 'Recording_day7_cafftwo_nopedestal';
basedir = ['.' filesep dirname filesep];
mdir = [basedir dirname '_matchedframes.mat'];
%% load
load(mdir)
num_markers = 20; %hard-coded for mouse
frame_period = 33.3333; %this should be in whatever units the matched frame files is in.
% Typically milliseconds.
%% This saves a data structure for use by Keras. 
for camerause = 1:numel(camnames)
    
    frame_inds = 1:frame_period:length(matched_frames_aligned{camerause});
    frame_inds = int32(round(frame_inds));
    
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

save([basedir camnames{camerause} '_MatchedFrames'],'data_frame','data_2d','data_3d','data_sampleID');

end
