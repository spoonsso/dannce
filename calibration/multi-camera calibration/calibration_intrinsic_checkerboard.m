%% Find the camera intrinsic parameters
% Input Parameters
clear
basedir = 'D:\20191122\mouse\calibration\intrinsic\';
cd(basedir)
numcams = 6;
squareSize = 10.0; % Size of Checkerboard squares in mm
ext = '.mp4';
maxNumImages = 500;
videoName = '0';
%% Automated Checkerboard Frame Detection
% Pre-allocate
params_individual = cell(1,numcams);
estimationErrors = cell(1,numcams);
imagePoints = cell(1,numcams);
boardSize = cell(1,numcams);
imagesUsed = cell(1,numcams);
imageNums = cell(1,numcams);

clear video_temp
for kk = 1:numcams
    
    tic
    video_temp = VideoReader([basedir filesep 'Camera' num2str(kk) filesep videoName '.mp4']);
    maxFrames = floor(video_temp.FrameRate*video_temp.Duration);
    
    video_base = cell(maxFrames,1);
    cnt = 1;
    while hasFrame(video_temp)
        video_base{cnt} = readFrame(video_temp,'native');
        cnt = cnt + 1;
    end
    num2use = length(video_base);
    clear video_temp
        
    imUse1 = round(linspace(1,length(video_base),num2use));
    fprintf('finding checkerboard points for view %i \n', kk)
    [imagePoints{kk}, boardSize{kk}, imagesUsed{kk}] = ...
        detectCheckerboardPointsPar(cat(4,video_base{imUse1}));

    worldPoints = generateCheckerboardPoints(boardSize{kk},squareSize);
    imagesUsedTemp = find(imagesUsed{kk});
    numImagesToUse = min([maxNumImages numel(imagesUsedTemp)]);
    [~,imageNums{kk}] = datasample(imagesUsedTemp,numImagesToUse,'Replace',false);
    disp(['Images used for view ' num2str(kk) ': ' num2str(numel(imageNums{kk}))]);
    I = video_base{1};
    imageSize = [size(I,1),size(I,2)];
    [params_individual{kk},pairsUsed,estimationErrors{kk}] = estimateCameraParametersPar(imagePoints{kk}(:,:,imageNums{kk}),worldPoints, ...
        'ImageSize',imageSize,'EstimateSkew',true,'EstimateTangentialDistortion',true,...
        'NumRadialDistortionCoefficients',3);
    toc
end
% Save the camera parameters
save([basedir 'cam_intrinsics.mat'],'params_individual','imagePoints','boardSize','imagesUsed','imageNums');

%% Visualize Preprojections
cd(basedir)
load('cam_intrinsics.mat')
numcams = 6;
for kk = 1:numcams
    video_temp = VideoReader([basedir 'view_cam' num2str(kk) '.mp4']);    
    maxframes = floor(video_temp.FrameRate*video_temp.Duration);
    video_base = cell(maxframes,1);
    cnt = 1;
    while hasFrame(video_temp)
        video_base{cnt} = readFrame(video_temp,'native');
        cnt = cnt + 1;
    end
    
    clear M
    figure;
%     imagesUsed_ = find(imagesUsed{kk});
    imagesUsed_ = imageNums{kk};
    imagesUsedFull_ = find(imagesUsed{kk});
    imagesUsedFull_ = imagesUsedFull_(imagesUsed_);
    
    for im2use = 1:numel(imagesUsed_)
        imUsed = imagesUsed_(im2use);
        imDisp = imagesUsedFull_(im2use);
        pts = imagePoints{kk}(:,:,imUsed);
        repro = params_individual{kk}.ReprojectedPoints(:,:,im2use);
        imagesc(video_base{imDisp});colormap(gray)
        hold on;
        plot(pts(:,1),pts(:,2),'or');
        plot(repro(:,1),repro(:,2),'xg');
        drawnow;
        M(im2use) = getframe(gcf);
    end
    
    % write reproject video
    %vidfile = [basedir 'reproject_view' num2str(kk) '.mp4'];
    %vk = VideoWriter(vidfile);
    %vk.Quality = 100;
    %open(vk)
    %writeVideo(vk,M);
    %close(vk);
    
end
%% View Undistorted Images
load([basedir 'cam_intrinsics.mat'])
for kk=1:numcams
    imFiles1 = VideoReader([basedir filesep 'Camera' num2str(kk) filesep '1' ext],'CurrentTime',0.5); 
    figure(kk);
    im = readFrame(imFiles1,'native');
    subplot(121);imagesc(im);
    subplot(122);imagesc(undistortImage(im,params_individual{kk}));
end























