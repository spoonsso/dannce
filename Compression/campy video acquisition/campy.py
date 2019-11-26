# Multiprocessing Pool method sustainable up to 120 Hz

import pypylon.pylon as pylon; import pypylon.genicam as geni
import numpy as np
import os
import time
from imageio import get_writer
from multiprocessing import Pool

# User-set Parameters
recdate = '20191122'
mouseid = 'mouse'
dirname = 'raw' # 'raw' 'videos' 'calibration\\intrinsic'
fname = '1' # name of video in each camera folder

frameRate = 100
recTimeInSec = 600
chunkSize = 3000
maxCams = 6
quality = '19'
timeout = 0

# Nvidia imposes a limit* of 2 encoding streams for consumer cards (GTX, most RTX, and some Quadro cards)
# *Limit is N+2 (N unrestricted + 2 total restricted streams) i.e. limit for 2x RTX 2070 cards is still 2 streams!
# assigns ffmpeg pipes to different NVENC chips (e.g. 0,1,2)
#gpuToUse = [1, 2, 1, 2, 1, 2] # GPUs 1 and 2 can easily keep up with 6 streams
gpuToUse = [2, 1, 0, 2, 1, 0] # Use GPUs on all 3 for now
ext = '.mp4'
base_folder_name = os.path.join('D:\\', recdate, mouseid, dirname, 'Camera')

# Automatic parameters
countOfImagesToGrab = recTimeInSec*frameRate

# User-Defined Functions
def OpenCamera(c):
    fileLoc = "C:\\Users\\Wang Lab\\Documents\\Basler\\Pylon5_settings"
    # color up to 100 Hz with no dropped frames
    nodeFile = "acA1920-150uc_1152x1024p_100fps_trigger_RGB_p6.pfs"
    os.chdir(fileLoc)

    # Open and load features for all cameras
    tlFactory = pylon.TlFactory.GetInstance()
    devices = tlFactory.EnumerateDevices()
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(devices[c]))
    serial = devices[c].GetSerialNumber()
    camera.Close()
    camera.StopGrabbing()
    camera.Open()
    pylon.FeaturePersistence.Load(nodeFile, camera.GetNodeMap(), True)
    print("Started camera", c, "serial#", serial)

    # Start grabbing frames (OneByOne = first in, first out)
    camera.MaxNumBuffer = 500
    camera.MaxNumGrabResults = 500
    camera.MaxNumQueuedBuffer = 500
    camera.OutputQueueSize = 500
    camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
    
    return camera

def StartPipe(c):
    gpuLabel = str(gpuToUse[c])
    file_name = base_folder_name + str(c+1) + os.sep + fname  + ext
    writer = get_writer(
        file_name, 
        fps = frameRate, 
        codec = 'h264_nvenc',  # H.264 hardware accelerated (GPU) 'h264_nvenc' 'hevc_nvenc'
        quality = None,  # disables variable compression
        pixelformat = 'bgr0',  # keep it as RGB colours
        ffmpeg_log_level = 'quiet', # 'warning', 'quiet', 'info'
        ffmpeg_params = [ # compatibility with older library versions
            '-preset', 'fast', # set to fast or llhq ('low latency' for h264 only)
            '-qp', quality,     # quality; 0 for lossless, 21 for "visually lossless"
            '-pix_fmt', 'bgr0',
            '-bf:v', '0',
            '-gpu', gpuLabel])
    print('Opened:', file_name, 'using GPU', gpuLabel)
    return writer
    
def GrabCam(camera,writer,c):
    c_count = 0
    grabtimes = []
    fnum = []

    try:
        while(True):
            try:
                # grab image from camera if waiting
                grabResult = camera.RetrieveResult(timeout, pylon.TimeoutHandling_ThrowException)
            
                if c_count is 0:
                    timeFirstGrab = grabResult.TimeStamp
                grabtime = (grabResult.TimeStamp - timeFirstGrab)/1e9
                grabtimes.append(grabtime)
                img = grabResult.Array
                grabResult.Release()

                # compress image and append to video file
                writer.append_data(img)

                c_count+=1
                fnum.append(c_count) # first frame = 1
                if c_count % chunkSize is 0:
                    print('Camera %i collected %i frames.' %(c,c_count))

            except geni.GenericException:
                time.sleep(0.000001)

            # To do: save timestamps with each frame number
            if not c_count in range(0,countOfImagesToGrab):
                break
    except KeyboardInterrupt:
        pass

    return c_count, fnum, grabtimes

def CloseCamera(camera, writer, c, c_count):
    camera.StopGrabbing()
    camera.Close()
    writer.close()    
    print('Camera', c, 'saved', str(c_count), 'frames')

def main(c):
    # Open camera(s)
    camera = OpenCamera(c)
    writer = StartPipe(c)   # Initialize ffmpeg pipe
    c_count, fnum, grabtimes = GrabCam(camera, writer, c) # Start retrieving frames
    CloseCamera(camera, writer, c, c_count)  # Closes cameras after acquisition
    x = np.array([fnum, grabtimes])
    fname = base_file_name + str(c+1) + os.sep + 'f' #.npy
    np.save(fname,x)

if __name__ == '__main__':
    # pool size >= maxCams (i.e. 6 parallel processes for 6 cameras)
    if maxCams > 1:
        p = Pool(maxCams) 
        p.map(main, range(0,maxCams))
    elif maxCams is 1:
        main(0)
