function [imagePoints, boardSize, imageIdx, userCanceled] = detectCheckerboardPointsPar(I, varargin)
% detectCheckerboardPoints Detect a checkerboard pattern in images
%   detectCheckerboardPoints can detect the keypoints of a checkerboard
%   calibration pattern in a single image, a set of images, or stereo image
%   pairs. In order to be detected, the size of the checkerboard must be at
%   least 4-by-4 squares.
%
%   [imagePoints, boardSize] = detectCheckerboardPoints(I) detects a checkerboard
%   in a 2-D truecolor or grayscale image I. imagePoints is an M-by-2 matrix 
%   of x-y coordinates of the corners of checkerboard squares. boardSize 
%   specifies the checkerboard dimensions as [rows, cols] measured in
%   squares. The number of points M is prod(boardSize-1). If the 
%   checkerboard cannot be detected, imagePoints = [] and boardSize = [0, 0].
%
%   [imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(imageFileNames)
%   detects a checkerboard pattern in images specified by imageFileNames 
%   cell array. imagePoints is an M-by-2-by-numImages array of x-y coordinates, 
%   where numImages is the number of images in which the checkerboard was 
%   detected. imagesUsed is a logical vector of the same size as 
%   imageFileNames. A value of true indicates that the pattern was detected
%   in the corresponding image.
%
%   [imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(images)
%   detects a checkerboard pattern in H-by-W images stored in an 
%   H-by-W-by-numColorChannels-by-numImages array. 
%
%   [imagePoints, boardSize, pairsUsed] = detectCheckerboardPoints(imageFileNames1,
%   imageFileNames2) detects a checkerboard pattern in stereo pairs of images
%   specified by imageFileNames1 and imageFileNames2 cell arrays. A value 
%   of true in the logical vector pairsUsed indicates that the checkerboard
%   was detected in the corresponding pair. imagePoints is an 
%   M-by-2-by-numPairs-by-numCameras array of x-y coordinates. 
%   imagePoints(:,:,:,1) are the points from the first set of images, and 
%   imagePoints(:,:,:,2) are the points from the second one. 
%
%   [imagePoints, boardSize, pairsUsed] = detectCheckerboardPoints(images1, images2)
%   detects a checkerboard pattern in stereo pairs of H-by-W images stored
%   in H-by-W-by-numColorChannels-by-numImages arrays.
%
%   [...] = detectCheckerboardPoints(..., Name, Value) specifies additional
%   name-value pair arguments described below:
% 
%   'MinCornerMetric'       Nonnegative scalar to specify the threshold for
%                           corner metric. If the image is noisy or highly
%                           textured, increase this threshold to reduce the
%                           number of false corners.
%  
%                           Default: 0.15
%  
% Class Support
% -------------
% I, images, images1, and images2 can be uint8, int16, uint16, single, or double.
% imageFileNames, imageFileNames1, and imageFilenames2 must be cell arrays of 
% strings.
%
% Example 1: Detect a Checkerboard in One Image
% ---------------------------------------------
% % Load an image containing the checkerboard pattern
% imageFileName = fullfile(toolboxdir('vision'),...
%         'visiondata', 'calibration', 'webcam', 'image4.tif');
% I = imread(imageFileName);
%
% % Detect the checkerboard points
% [imagePoints, boardSize] = detectCheckerboardPoints(I);
%
% % Display detected points
% J = insertText(I, imagePoints, 1:size(imagePoints, 1));
% J = insertMarker(J, imagePoints, 'o', 'Color', 'red', 'Size', 5);
% imshow(J);
% title(sprintf('Detected a %d x %d Checkerboard', boardSize));
%
% Example 2: Detect Checkerboard in a Set of Image Files
% ------------------------------------------------------
% % Create a cell array of file names of calibration images
% for i = 1:5
%     imageFileName = sprintf('image%d.tif', i);
%     imageFileNames{i} = fullfile(toolboxdir('vision'),...
%          'visiondata', 'calibration', 'webcam', imageFileName);
% end
%
% % Detect calibration pattern
% [imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(imageFileNames);
%
% % Display detected points
% imageFileNames = imageFileNames(imagesUsed);
% for i = 1:numel(imageFileNames)
%     I = imread(imageFileNames{i});
%     subplot(2, 2, i);
%     imshow(I); hold on; plot(imagePoints(:,1,i), imagePoints(:,2,i), 'ro');
% end
%
% Example 3: Detect Checkerboard in Stereo Images
% -----------------------------------------------
% % Specify calibration images
% imageDir = fullfile(toolboxdir('vision'), 'visiondata', ...
%     'calibration', 'stereo');
% leftImages = imageDatastore(fullfile(imageDir, 'left'));
% rightImages = imageDatastore(fullfile(imageDir, 'right'));
% images1 = leftImages.Files;
% images2 = rightImages.Files;
%
% % Detect the checkerboards
% [imagePoints, boardSize, pairsUsed] = detectCheckerboardPoints(images1,...
%   images2);     
%
% % Display points from first 4 camera 1 images
% images1 = images1(pairsUsed);
% figure;
% for i = 1:4
%     I = imread(images1{i});
%     subplot(2, 2, i);
%     imshow(I); hold on; plot(imagePoints(:,1,i,1), imagePoints(:,2,i,1), 'ro');
% end 
% annotation('textbox', [0 0.9 1 0.1], 'String', 'Camera 1', ...
%    'EdgeColor', 'none', ...
%    'HorizontalAlignment', 'center')
%
% % Display points from first 4 camera 2 images
% images2 = images2(pairsUsed);
% figure;
% for i = 1:4
%     I = imread(images2{i});
%     subplot(2, 2, i);
%     imshow(I); hold on; plot(imagePoints(:,1,i,2), imagePoints(:,2,i,2), 'ro');
% end 
% annotation('textbox', [0 0.9 1 0.1], 'String', 'Camera 2', ...
%    'EdgeColor', 'none', ...
%    'HorizontalAlignment', 'center')
% 
% See also estimateCameraParameters, generateCheckerboardPoints,
%   cameraCalibrator, cameraParameters, stereoParameters

% Copyright 2013-2018 MathWorks, Inc.

% References:
% -----------
% Andreas Geiger, Frank Moosmann, Omer Car, and Bernhard Schuster, 
% "Automatic Camera and Range Sensor Calibration using a single Shot. 
% In International Conference on Robotics and Automation (ICRA), St. Paul, 
% USA, May 2012.

%#codegen
%#ok<*EMCLS>
%#ok<*EMCA>

if isempty(coder.target)
    % Convert strings to chars for simulation
    
    if isstring(I)
        I = convertStringsToChars(I);
    end
    
    if nargin>1
        [varargin{:}] = convertStringsToChars(varargin{:});
    end
    
    [images2, showProgressBar, minCornerMetric] = parseInputs(varargin{:});
else
    coder.internal.errorIf(ischar(I), 'vision:calibrate:codegenFileNamesNotSupported');
    coder.internal.errorIf(isstring(I), 'vision:calibrate:codegenFileNamesNotSupported');
    coder.internal.errorIf(iscell(I), 'vision:calibrate:codegenFileNamesNotSupported');
    coder.internal.errorIf(isnumeric(I) && size(I, 4) > 1,...
        'vision:calibrate:codegenMultipleImagesNotSupported');
    [images2, showProgressBar,minCornerMetric] = parseInputsCodegen(varargin{:});
end

if isempty(images2) 
    % single camera
    [imagePoints, boardSize, imageIdx, userCanceledTmp] = detectMono(I, ...
        showProgressBar, minCornerMetric);
else
    % 2-camera stereo 
    images1 = I;
    checkStereoImages(images1, images2);
    [imagePoints, boardSize, imageIdx, userCanceledTmp] = detectStereo(images1, ...
        images2, showProgressBar, minCornerMetric);
end

checkThatBoardIsAsymmetric(boardSize);
if showProgressBar
    userCanceled = userCanceledTmp;
else
    userCanceled = false;
end

%--------------------------------------------------------------------------
function [image2, showProgressBar, minCornerMetric] = parseInputs(varargin)

% Check if the second argument is the second set of images
% Need to do this "by hand" because inputParser does not 
% handle optional string arguments.

isSecondArgumentString =  ~isempty(varargin) && ischar(varargin{1});

isSecondArgumentNameValuePair = false;
if isSecondArgumentString
    % pv-pairs comes in pairs
    isSecondArgumentNameValuePair = (bitget(numel(varargin), 1) == 0);
end

if isempty(varargin) || isSecondArgumentNameValuePair
    image2 = [];
    args = varargin;
else
    image2 = varargin{1};
    if numel(varargin) > 1
        args = varargin(2:end);
    else
        args = {}; 
    end
end

% Parse the Name-Value pairs
parser = inputParser;
parser.addParameter('ShowProgressBar', false, @checkShowProgressBar);
parser.addParameter('MinCornerMetric', 0.15, @checkMinCornerMetric);
parser.parse(args{:}); 
showProgressBar = parser.Results.ShowProgressBar;
minCornerMetric = parser.Results.MinCornerMetric;

%--------------------------------------------------------------------------
function [image2, showProgressBar, minCornerMetric] = parseInputsCodegen(varargin)

showProgressBar = false;

isSecondArgumentString =  ~isempty(varargin) && ischar(varargin{1});

isSecondArgumentNameValuePair = false;
if isSecondArgumentString
    % pv-pairs comes in pairs
    isSecondArgumentNameValuePair = (bitget(numel(varargin), 1) == 0);
end

if isempty(varargin) || isSecondArgumentNameValuePair
    image2 = [];
    args = varargin;
else
    image2 = varargin{1};
    if numel(varargin) > 1
        args = varargin(2:end);
    else
        args = {}; 
    end
end

if ~isempty(args)
    params = struct( ...
        'MinCornerMetric', uint32(0));
    
    popt = struct( ...
        'CaseSensitivity', false, ...
        'StructExpand',    true, ...
        'PartialMatching', true);
    
    optarg = eml_parse_parameter_inputs(params, popt, args{:});
    minCornerMetric = eml_get_parameter_value(optarg.MinCornerMetric, 0.15, args{:});
        
    % MinCornerMetric    
    checkMinCornerMetric(minCornerMetric);        
else
    minCornerMetric = 0.15;
end

%--------------------------------------------------------------------------
% Detect the checkerboards in a single set of images
function [points, boardSize, imageIdx, userCanceled] = ...
    detectMono(I, showProgressBar, minCornerMetric)

userCanceled = false;
if iscell(I)
    % detect in a set of images specified by file names
    fileNames = I;
    checkFileNames(fileNames);    
    [points, boardSize, imageIdx, userCanceledTmp] = ...
        detectCheckerboardFiles(fileNames, showProgressBar, minCornerMetric);
    if showProgressBar
        userCanceled = userCanceledTmp;
    end
elseif ischar(I)
    % detect in a single image specified by a file name
    fileName = I;
    checkFileName(I);
    I = imread(fileName);
    [points, boardSize] = detectCheckerboardInOneImage(I, minCornerMetric);
    imageIdx = ~isempty(points);
elseif ndims(I) > 3
    % detect in a stack of images
    checkImageStack(I);
    [points, boardSize, imageIdx, userCanceledTmp] = ...
        detectCheckerboardStack(I, showProgressBar, minCornerMetric);
    if showProgressBar
        userCanceled = userCanceledTmp;
    end    
else
    % detect in a single image
    checkImage(I);
    [points, boardSize] = detectCheckerboardInOneImage(I, minCornerMetric);
    imageIdx = ~isempty(points);
end

%--------------------------------------------------------------------------
% Detect the checkerboards in stereo pairs.
function [points, boardSize, imageIdx, userCanceled] = ...
    detectStereo(images1, images2, showProgressBar, minCornerMetric)

if isnumeric(images1) && size(images1, 4) == 1 % pair of single images
    [points1, boardSize1] = detectMono(images1, false, minCornerMetric);
    [points2, boardSize2] = detectMono(images2, false, minCornerMetric);
    
    userCanceled = false;
    if ~isequal(boardSize1, boardSize2)
        points = zeros(0, 2);
        boardSize = [0,0];
        imageIdx = false;
    else
        points = cat(4, points1, points2);
        boardSize = boardSize1;
        imageIdx = true;
    end
    
    if isempty(points)
        imageIdx = false;
    end
else
    % concatenate the two sets of images into one
    images = concatenateImages(images1, images2);
    
    % detect the checkerboards in the combined set
    [points, boardSize, imageIdx, userCanceled] = detectMono(images, ...
        showProgressBar, minCornerMetric);
    
    if userCanceled
        points = zeros(0, 2);
        boardSize = [0,0];
    else
        % separate the points from images1 and images2
        [points, imageIdx] = separatePoints(points, imageIdx);
        
        if isempty(points)
            boardSize = [0 0];
        end
    end
end

%--------------------------------------------------------------------------
function images = concatenateImages(images1, images2)
if iscell(images1)
    images = {images1{:}, images2{:}}; %#ok
elseif ischar(images1)
    images = {images1, images2}; 
else
    images = cat(4, images1, images2);
end

%--------------------------------------------------------------------------
function [points, imageIdx] = separatePoints(points, imageIdx)

numImages = numel(imageIdx);

% get the indices corresponding to the two original sets
leftRightIdx = ones(numImages, 1);
leftRightIdx(end/2+1:end) = 2;

% find the pairs where the checkerboard was detected in both images
imagesUsedLeft  = imageIdx(1:end/2);
imagesUsedRight = imageIdx(end/2+1:end);
commonImagesIdx = imagesUsedLeft & imagesUsedRight;
commonImagesIdxFull = [commonImagesIdx; commonImagesIdx];

% get points1
pointsLeftIdx = commonImagesIdxFull & (leftRightIdx == 1);
pointsLeftIdx = pointsLeftIdx(imageIdx);
pointsLeft = points(:, :, pointsLeftIdx);

% get points2
pointsRightIdx = commonImagesIdxFull & (leftRightIdx == 2);
pointsRightIdx = pointsRightIdx(imageIdx);
pointsRight = points(:, :, pointsRightIdx);

% combine the points into a 4D array
points = cat(4, pointsLeft, pointsRight);
imageIdx = commonImagesIdx;

%--------------------------------------------------------------------------
function tf = checkShowProgressBar(showProgressBar)
validateattributes(showProgressBar, {'logical', 'numeric'},...
    {'scalar'}, mfilename, 'ShowProgressBar'); 
tf = true;

%--------------------------------------------------------------------------
function tf = checkMinCornerMetric(value)
validateattributes(value, {'single', 'double'},...
    {'scalar', 'real', 'nonnegative', 'finite'}, mfilename, 'MinCornerMetric');
tf = true;

%--------------------------------------------------------------------------
function checkImage(I)
vision.internal.inputValidation.validateImage(I, 'I');

%--------------------------------------------------------------------------
function checkImageStack(images)
validClasses = {'double', 'single', 'uint8', 'int16', 'uint16'}; 
validateattributes(images, validClasses,...
    {'nonempty', 'real', 'nonsparse'},...
    mfilename, 'images'); 
coder.internal.errorIf(size(images, 3) ~= 1 && size(images, 3) ~= 3,...
    'vision:dims:imageNot2DorRGB');

%--------------------------------------------------------------------------
function checkFileNames(fileNames)
validateattributes(fileNames, {'cell'}, {'nonempty', 'vector'}, mfilename, ...
    'imageFileNames'); 
for i = 1:numel(fileNames)
    checkFileName(fileNames{i}); 
end

%--------------------------------------------------------------------------
function checkFileName(fileName)
validateattributes(fileName, {'char'}, {'nonempty'}, mfilename, ...
    'elements of imageFileNames'); 

try %#ok<EMTC>
    state = warning('off','imageio:tifftagsread:badTagValueDivisionByZero');
    imfinfo(fileName);
catch e
    warning(state);
    throwAsCaller(e);
end
warning(state);

%--------------------------------------------------------------------------
function checkStereoImages(images1, images2)
coder.internal.errorIf(strcmp(class(images1), class(images2)) == 0,...
    'vision:calibrate:stereoImagesMustBeSameClass');

coder.internal.errorIf(~ischar(images1) && any(size(images1) ~= size(images2)),...
    'vision:calibrate:stereoImagesMustBeSameSize');

%--------------------------------------------------------------------------
function checkThatBoardIsAsymmetric(boardSize)
% ideally, a board should be asymmetric: one dimension should be even, and
% the other should be odd.
if isempty(coder.target)
    if ~all(boardSize == 0) && (~xor(mod(boardSize(1), 2), mod(boardSize(2), 2))...
            || boardSize(1) == boardSize(2))
        s = warning('query', 'backtrace');
        warning off backtrace;
        warning(message('vision:calibrate:boardShouldBeAsymmetric'));
        warning(s);
    end
end

%--------------------------------------------------------------------------
% Detect checkerboards in a set of images specified by file names
function [points, boardSize, imageIdx, userCanceled] = ...
    detectCheckerboardFiles(fileNames, showProgressBar, minCornerMetric)
numImages = numel(fileNames);
boardPoints = cell(1, numImages);
boardSizes = zeros(numImages, 2);
userCanceled = false;
if showProgressBar
    waitBar = ...
        vision.internal.calibration.checkerboard.DetectionProgressBar(numImages);
end
for i = 1:numImages
    if showProgressBar && waitBar.Canceled
            points = [];
            boardSize = [0 0];
            imageIdx =[];
            userCanceled = true;
            return;
    end

    im = imread(fileNames{i}); 
    [boardPoints{i}, boardSizes(i,:)] = detectCheckerboardInOneImage(im, minCornerMetric);     
    if showProgressBar
        waitBar.update();
    end    
end
[points, boardSize, imageIdx] = chooseValidBoards(boardPoints, boardSizes);

%--------------------------------------------------------------------------
% Detect checkerboards in a stack of images
function [points, boardSize, imageIdx, userCanceled] = ...
    detectCheckerboardStack(images, showProgressBar, minCornerMetric)
numImages = size(images, 4);
boardPoints = cell(1, numImages);
boardSizes = zeros(numImages, 2);
userCanceled = false;
% if showProgressBar
%     waitBar = ...
%         vision.internal.calibration.checkerboard.DetectionProgressBar(numImages);
% end
parfor i = 1:numImages
%     if showProgressBar && waitBar.Canceled
%             points = [];
%             boardSize = [0 0];
%             imageIdx =[];
%             userCanceled = true;
%             return;
%     end    
    im = images(:, :, :, i);
    [boardPoints{i}, boardSizes(i,:)] = detectCheckerboardInOneImage(im, minCornerMetric); 
%     if showProgressBar
%         waitBar.update();
%     end        
end
[points, boardSize, imageIdx] = chooseValidBoards(boardPoints, boardSizes);

%--------------------------------------------------------------------------
% Determine which board size is the most common in the set
function [points, boardSize, imageIdx] = chooseValidBoards(boardPoints, boardSizes)
uniqueBoardIds = 2.^boardSizes(:, 1) .* 3.^boardSizes(:, 2);

% Eliminate images where no board was detected.
% The unique board id in this case is 2^0 + 3^0 = 1.
% Replace all 1's by a sequence of 1:n * 1e10, which will be different from
% all other numbers which are only multiples of 2 and 3.
zeroIdx = (uniqueBoardIds == 1);
uniqueBoardIds(zeroIdx) = (1:sum(zeroIdx)) * 5;

% Find the most common value among unique board ids.
[~, ~, modes] = mode(uniqueBoardIds);
modeBoardId = max(modes{1}); 

% Get the corresponding points
imageIdx = (uniqueBoardIds == modeBoardId);
boardSize = boardSizes(imageIdx, :);
boardSize = boardSize(1, :);
points = boardPoints(imageIdx);
points = cat(3, points{:}); 

%--------------------------------------------------------------------------
function [points, boardSize] = detectCheckerboardInOneImage(Iin, minCornerMetric)
if ismatrix(Iin)
    Igray = Iin;    
else
    Igray = rgb2gray(Iin);
end
I = im2single(Igray);

% bandwidth to smooth the image
sigma = 2;

[points, boardSize] = vision.internal.calibration.checkerboard.detectCheckerboard(...
    I, sigma, minCornerMetric);

if isempty(points)
    sigma = 4;
    [points, boardSize] = vision.internal.calibration.checkerboard.detectCheckerboard(...
        I, sigma, minCornerMetric);
end

