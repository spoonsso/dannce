%% Given a full mocap_data structure, detects bad mocap frames (all x-y-z 0
% for a given marker at a single time point) and replaces them with NaN
function  data = correct_mocap(data)
    for ll =1:numel(data.markernames)
    nandata = data.markers_preproc.(data.markernames{ll});
    nandata(nandata(:,1)==0 & nandata(:,2)==0 & nandata(:,3)==0,:) = nan;
    data.markers_preproc.(data.markernames{ll}) = nandata;
    end
end