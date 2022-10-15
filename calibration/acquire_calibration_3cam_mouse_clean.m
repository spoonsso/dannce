%% Script for calibrating multiple cameras in Matlab
% Jesse Marshall 2019 

%acquire calibration info
imaqreset
numcams = 3;
vid = cell(1,numcams);
logfile = cell(1,numcams);

%% ENTER PARENT PATH HERE
parentpath = 'C:\RatControl\Matlab_camera_calibrate';

hires =1;
numframes_aq = 100;

logfile_tag = 'videofiles_run1';
lframe_tag = 'lframe_labels';
worldcoord_tag = 'worldcoordinates_lframe';
%% ENTER SIZE OF LFRAME HERE
LFrame_coordinates = [0 0 22; 30 0 22; 90 0 22; 0 60 22];

lframename = strcat(parentpath,filesep,lframe_tag,'.mat');
lframename_checkerboard = strcat(parentpath,filesep,'lframe_labels_checkboard','.mat');

worldcoordinates_framename = strcat(parentpath,filesep,worldcoord_tag,'.mat');
worldcoordinates_framename_2 = strcat(parentpath,filesep,worldcoord_tag,'_2.mat');

logfile_names = cell(1,numcams);
mkdir(parentpath);
squareSize = 16.35;
for kk = 1:numcams
    logfile_names{kk} = strcat(parentpath,filesep,logfile_tag,num2str(kk),'.avi');
end

for kk = 1:numcams
    if hires ==1
    vid{kk} = videoinput('pointgrey', kk, 'F7_YUV422_1328x1048_Mode0');
    else
       vid{kk} = videoinput('pointgrey', kk, 'F7_YUV422_656x524_Mode4');
    end
    
    src = getselectedsource(vid{kk});
    vid{kk}.FramesPerTrigger = 1;
    vid{kk}.TriggerRepeat = 1000;
    
    triggerconfig(vid{kk}, 'manual')
    
    vid{kk}.ReturnedColorspace = 'rgb';
    set(vid{kk}, 'LoggingMode', 'Disk&Memory');
    logfile_names{kk} = strcat(parentpath,filesep,logfile_tag,num2str(kk),'.avi');
    logfile{kk} = VideoWriter(logfile_names{kk});
    
    vid{kk}.DiskLogger = logfile{kk};
    start(vid{kk})
end
%preview(vid{1});


%% get matched frames from the cameras wth
%% STEP 1- CHECKERBOARD
fprintf('starting in 10 s')
pause(10)

%in X Y Z, 1 23 are in a line
fprintf('starting now')

for ll =1:numframes_aq
    fprintf('triggering \n')
    for kk = 1:numcams
    trigger(vid{kk})
    end
    
    pause(0.5)
end

%% acquire and save postion of markers
%% STEP 2 (Put L-frame in arena first)

input('Hit enter when the grid is in the arena \n')
%fprintf('enter or double click to stop labeling points \n')
frame_image = cell(1,numcams);

for kk = 1:numcams    
     frame_image{kk} = getsnapshot(vid{kk});
end

    save(lframename,'frame_image');
%     save('testimages','frame_image');

    
    %% STEP 2.5 (Put checkerboard in arena close to CameraR and have one light on)

input('Hit enter when the checkerboard is in the arena \n')
%fprintf('enter or double click to stop labeling points \n')
checkerboard_frame_image = cell(1,numcams);

for kk = 1:numcams    
     checkerboard_frame_image{kk} = getsnapshot(vid{kk});
end

    save(lframename_checkerboard,'checkerboard_frame_image');
    
    
%save images and points
%% clear these to facilitate clearing them out
%% STEP 3 -- run through lframe

for kk = 1:numcams
    delete(vid{kk})
    delete(logfile{kk})
end




%% get orientation from mocap arena
 %% get single camera calibrations
 params_individual=cell(1,numcams);
 estimationErrors = cell(1,numcams);
    for kk = [1:numcams]%1:numcams
        tic
      imFiles1 = VideoReader(logfile_names{(kk)});    
    video_base = cell(1,0);
    
    
    while hasFrame(imFiles1)
        video_base{end+1} = readFrame(imFiles1);
    end
        num2use = length(video_base);

        imUse1 = round(linspace(1,length(video_base),num2use));
     fprintf('finding checkerboard points \n')
    [imagePoints, boardSize, imagesUsed] = ...
        detectCheckerboardPoints(cat(4,video_base{imUse1}));
  %  squareSize = 16.57; %mm
    
    worldPoints = generateCheckerboardPoints(boardSize,squareSize);
    imagesUsed = find(imagesUsed);
    fprintf('images used %f \n',numel(imagesUsed))
    I = video_base{1};%readimage(imFiles2,1);
    imageSize = [size(I,1),size(I,2)];
     [params_individual{kk},pairsUsed,estimationErrors{kk}] = estimateCameraParameters(...
         imagePoints,worldPoints, ...
        'ImageSize',imageSize,...
        'EstimateTangentialDistortion',true,...
        'NumRadialDistortionCoefficients',3,'EstimateSkew',true);
    figure(880+kk)
    showReprojectionErrors(params_individual{kk});
print('-dpng',strcat(parentpath,filesep,'Reprojection_errors_cam',num2str(kk),'.png'))
    
    
    figure(980+kk) 
imshow(video_base{imagesUsed(1)}); 
hold on;
plot(imagePoints(:,1,1), imagePoints(:,2,1),'go');
plot(params_individual{kk}.ReprojectedPoints(:,1,1),...
    params_individual{kk}.ReprojectedPoints(:,2,1),'r+');
legend('Detected Points','ReprojectedPoints');
print('-dpng',strcat(parentpath,filesep,'Reprojection_image_cam',num2str(kk),'.png'))

hold off;
    toc
    
    fprintf('camera parameters \n')
        params_individual{kk}

    
    end
    
    
    
    %end of step 3
    %%-----------------------------------
    %% lframe analysis
    
    %% STEP 4 - label lframe
fprintf('starting lframe analysis\n')
%cam1 =num2str(basecam');
%cam2 = {'2','2','3','4','5','6'};
%params_comp = cell(1,numel(cam1));
%params_comp{kk} = params;
point_coordinates = cell(1,numcams);
load(lframename)
for kk = [1:numcams]%1:numcams
    %load camera parameters
    %load(fullfile(parentpath,strcat(logfile_tag,'_CalibImgs',[cam1(kk) '-' cam2{kk} '.mat'])));
camparams =params_individual{kk};
     %frame = getsnapshot(vid{kk});
  figure(8077)
     LFrame_image{kk} = undistortImage(frame_image{kk},camparams);
image(LFrame_image{kk})
pause(8)
[xi,yi] = getpts ;
point_coordinates{kk} = [xi yi];
end
for kk = 1:numcams
point_coordinates{kk} = reshape(point_coordinates{kk},[],2);
point_coordinates{kk} =point_coordinates{kk}(1:4,:);
    end

    save(lframename,'frame_image','LFrame_image','point_coordinates','params_individual','estimationErrors');
    
    
    %% Step 5 compute the world orientation
    load(lframename)
worldOrientation = cell(1,numcams);
    worldLocation = cell(1,numcams);
    rotationMatrix = cell(1,numcams);
    translationVector = cell(1,numcams);
    %% get the orienation and location of cameras
    for kk = [1:numcams]%:numcams
      
     [worldOrientation{kk},worldLocation{kk}] = estimateWorldCameraPose(double(point_coordinates{kk}),...
         double(LFrame_coordinates),params_individual{kk},'Confidence',95,'MaxReprojectionError',4,'MaxNumTrials',5000);
     
     [rotationMatrix{kk},translationVector{kk}] = cameraPoseToExtrinsics(worldOrientation{kk},worldLocation{kk});
     
    if ismember(kk,[3,4,6])
     % worldLocation{kk}(2) = -worldLocation{kk}(2);
   %  if kk == 1
     % worldOrientation{kk} = worldOrientation{kk}';
     % end
    end
%      if ismember(kk,[2])
%      worldLocation{kk}(1:2) = -worldLocation{kk}(1:2);
%      worldOrientation{kk} = worldOrientation{kk}';
%    end
   
     figure(222)
    plotCamera('Location',worldLocation{kk},'Orientation',worldOrientation{kk},'Size',50,'Label',num2str(kk));
hold on
if (kk == numcams)
    print('-dpng',strcat(parentpath,filesep,'cameraarrangement.png'));
end
%[R,t] = extrinsics(double(point_coordinates{kk}),double(LFrame_coordinates(:,1:2)),camparams);

%% look at reprojection in image from estimated world pose
figure(233+kk)
image( LFrame_image{kk})
hold on

imagePoints = worldToImage(params_individual{kk},rotationMatrix{kk},translationVector{kk},double(LFrame_coordinates));

colorarray = {'r','g','b','y'};
for llll = 1:size(imagePoints,1)
plot(imagePoints(llll,1),imagePoints(llll,2),'o','MarkerSize',4,...
    'MarkerEdgeColor',colorarray{llll},'MarkerFaceColor',colorarray{llll})
end
hold off
    print('-dpng',strcat(parentpath,filesep,'camerareproject',num2str(kk),'.png'));

    end
fprintf('saving world coordinates \n')
   save( worldcoordinates_framename,'params_individual',...
       'worldOrientation','worldLocation','rotationMatrix','translationVector');

    
    
      %% STEP 6 - refine with checkerboard
fprintf('starting checkerboard analysis:\n MAKE SURE ALL POINTS ARE TRACKED AND RED POINT IS OPPOSITE THE BEND OF THE L AND BLUE POINT IS ON SHORT AXIS\n If this is a problem, know that you dont need to click on the L exactly, but can exaggerate the positions. In the code, consider making shortdim equal to the opposite of longdim')
%cam1 =num2str(basecam');
%cam2 = {'2','2','3','4','5','6'};
%params_comp = cell(1,numel(cam1));
%params_comp{kk} = params;

load(lframename_checkerboard)
allimagepoints = cell(1,numcams);
for kk = [1:numcams]%1:numcams
    camparams =params_individual{kk};

    undistorted_checkerboard =  undistortImage(checkerboard_frame_image{kk},camparams);
       
      fprintf('finding checkerboard points \n')
    [imagePoints_ch, boardSize_ch, imagesUsed] = ...
        detectCheckerboardPoints(undistorted_checkerboard,'MinCornerMetric',0.15);
    if size(imagePoints_ch,1)<40
       [imagePoints_ch, boardSize_ch, imagesUsed] = ...
        detectCheckerboardPoints(undistorted_checkerboard,'MinCornerMetric',0.15);
    end
    squareSize_ch = 16.57; %mm    
    worldPoints = generateCheckerboardPoints(boardSize_ch,squareSize_ch);
 figure(8078) 
imshow(undistorted_checkerboard); 
hold on;
[xi,yi] = getpts ;
point_coordinates_ch{kk} = [xi yi];
point_coordinates_ch{kk} = reshape(point_coordinates_ch{kk},[],2);
point_coordinates_ch{kk} =point_coordinates_ch{kk}(1:4,:);
origin = point_coordinates_ch{kk}(1,:);
long_axis = -(point_coordinates_ch{kk}(3,:)-point_coordinates_ch{kk}(1,:));
short_axis = -(point_coordinates_ch{kk}(4,:)-point_coordinates_ch{kk}(1,:));

detected_long_axis = -(imagePoints_ch(1,:)-imagePoints_ch(min(boardSize_ch),:));
detected_short_axis = -(imagePoints_ch(1,:)-imagePoints_ch(min(boardSize_ch)-1,:));

long_axis_dot = dot(long_axis,detected_long_axis)./(norm(long_axis)*norm(detected_long_axis));
short_axis_dot =dot(short_axis,detected_short_axis)./(norm(short_axis)*norm(detected_short_axis));
disp(long_axis_dot)
disp(short_axis_dot)

[~,longdim] = max(abs(long_axis)); 
[~,shortdim] = max(abs(short_axis)); 
flipped_undistorted_checkerboard=undistorted_checkerboard;
pointindex = 1:((boardSize_ch(1)-1)*((boardSize_ch(2)-1)));
pointindex = reshape(pointindex,min(boardSize_ch)-1,[]);

if long_axis_dot<0
    fprintf('flipped long \n')
    pointindex = flip(pointindex,longdim);
end

if short_axis_dot<0
    fprintf('flipped short \n')
    pointindex = flip(pointindex,shortdim);
end

pointindex_unflipped = reshape(pointindex,[],1);
%   [imagePoints_ch, boardSize_ch, imagesUsed] = ...
%         detectCheckerboardPoints(flipped_undistorted_checkerboard,'MinCornerMetric',0.15);
imagePoints_ch = imagePoints_ch(pointindex_unflipped,:);


figure(8078+kk) 
imshow(undistorted_checkerboard); 
hold on;
plot(imagePoints_ch(1,1), imagePoints_ch(1,2),'ro');
plot(imagePoints_ch(2,1), imagePoints_ch(2,2),'bo');
plot(imagePoints_ch(3:end,1), imagePoints_ch(3:end,2),'go');
hold off
%plot(imagePoints_ch(:,1), imagePoints_ch(:,2),'go');
    allimagepoints{kk} = imagePoints_ch;
    checkerboard_images_undistorted{kk} = undistorted_checkerboard;
    boardSize_ch_full{kk} = boardSize_ch;
end
    save(lframename_checkerboard,'checkerboard_images_undistorted','allimagepoints',...
        'params_individual','estimationErrors','boardSize_ch_full');

    
    
    %% STEP 7 compute the corrected world coordinates
for kk = 1:numcams
    camMatrix{kk} = cameraMatrix(params_individual{kk},rotationMatrix{kk},translationVector{kk});
end
   % camMatrix{1} = cameraMatrix(params_individual{1},rotationMatrix_2{1},translationVector_2{1});

minpts = min(size(allimagepoints{1},1),size(allimagepoints{2},1));
 point3d = triangulate(allimagepoints{2}(1:minpts,:), allimagepoints{3}(1:minpts,:),...
             camMatrix{2},camMatrix{3});
         
     for    kk=1:numcams
[X,Y] = meshgrid(1:(max(boardSize_ch_full{kk})-1),1:(min(boardSize_ch_full{kk})-1));
y_step = squareSize;
x_step = squareSize;
location3d = [(reshape(Y,[],1)-1)*y_step (reshape(X,[],1)-1)*x_step  zeros(numel(X),1)];

     [worldOrientation_2{kk},worldLocation_2{kk}] = estimateWorldCameraPose(double(allimagepoints{kk}),...
         double(location3d),params_individual{kk},'Confidence',95,'MaxReprojectionError',4,'MaxNumTrials',5000);
     
     [rotationMatrix_2{kk},translationVector_2{kk}] = cameraPoseToExtrinsics(worldOrientation_2{kk},worldLocation_2{kk});
     
figure(333+kk)
image( checkerboard_images_undistorted{kk})
hold on
imagePoints = worldToImage(params_individual{kk},rotationMatrix{kk},...
    translationVector{kk},double(point3d));
imagePoints2 = worldToImage(params_individual{kk},rotationMatrix_2{kk},...
    translationVector_2{kk},double(location3d));
plot(allimagepoints{kk}(:,1),allimagepoints{kk}(:,2),'or')
plot(imagePoints2(:,1),imagePoints2(:,2),'og')
plot(imagePoints(:,1),imagePoints(:,2),'ok')

hold off
legend('Ground truth','Checkerboard','LFrame')
    print('-dpng',strcat(parentpath,filesep,'camerareproject_checkerboard',num2str(kk),'.png'));

    

figure(227)
    plotCamera('Location',worldLocation_2{kk},'Orientation',worldOrientation_2{kk},'Size',50,'Label',num2str(kk));
hold on
view([-91 84])
if (kk == numcams)
    print('-dpng',strcat(parentpath,filesep,'cameraarrangement_2.png'));
end
%  
    
     end
     
fprintf('saving world coordinates \n')
   save( worldcoordinates_framename_2,'params_individual','worldOrientation_2','worldLocation_2','rotationMatrix_2','translationVector_2');

        
        
   
   
   
   
   