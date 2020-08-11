# Instructions for campy, a Basler + ffmpeg video acquisition pipeline
# My machine is Windows 10, i9-9900X on X299 mobo with Titan RTX and M60 GPUs
# Can sustain 24-bit color video acquisition from 6 cameras at 100 Hz
# File size ~170 GB/hour instead of 7.6 TB/hour (~45x reduction with qp=19)

#Hardware
# See Nvidia encode decode matrix for supported cards and number of encoding streams
GTX graphics card for up to 2 streams
Quadro P2000 or M4000 should work for 3 streams

#Software
Anaconda (Python 3.7)
Pylon 6 (USB3 developer version)
Nvidia Geforce experience (update drivers to >426.xx)

#Download and save campy.py in a directory "basedir"

#Python package install (open command line in admin mode)
conda install pypylon
conda install imageio-ffmpeg -c conda-forge

# Open Pylon 6 GUI, open camera and tune "features" for your Basler camera
# Suggestions (in Guru mode)
1) In Image Format Control, set:
	Width, Height = ROI, multiple of 128 (e.g. 1024,1152)
	Center x and y (move camera to capture arena)
	Pixel format = RGB8
2) In Image Quality Control, set:
	In PGI Control (non-Bayer only): 
		Demosaicing Mode = Basler PGI
		Noise Reduction = 0.0
		Sharpness Enhancement = 1.0
	Light Source Preset = Daylight (5000 or 6500 Kelvin)
	Balance White Control = Coninuous
3) In Acquisition Control, set:
	Shutter mode = Global
	Exposure Mode = Timed
	Exposure Time <= 2000 us or what lighting allows w/o saturation
	Sensor Readout Mode = Fast
	Trigger Mode = On (frame start on rising edge, Line 3 or Line 4)
	Enable Acqusition Frame Rate = True
	Acquisition Frame Rate = 1000000
	Resulting Frame Rate (Automatically calculated, mine ~107 fps, set trigger to 100.00 fps)
4) In Digital I/O Control, set:
	Line Selector = Line 4 (or Line not used to trigger)
	Line Mode = Output
	Line Source = Exposure Active
	Line Inverter = True
5) In Device Control, set:
	Device Link Throughput Limit Mode = Off

# Test your triggered frame rate is achieved without dropped frames using Bandwidth Manager, under Tools
	fps Received mark should be very close to your frame trigger rate
	Bandwith should be >355 MB/s for each USB controller
	You may need a PCie USB expansion card to break out more high bandwidth controllers than your motherboard's I/O allows

# Save .pfs feature file in a directory, such as basedir 

# Edit script for your camera parameters
frameRate (same as hardware trigger frequency. Set below the maximum frame rate Pylon Viewer says under Acquisition)
nodeFile and fileLoc (feature file location and name)
gpuToUse (0 is first GPU, 1 is second, etc.)
maxCams = 3 (or number of cameras you have)
file name (I will write GUI wrapper to make this easier in the future)
quality = '19' (ffmpeg compression level '0' is no compression, higher number is higher compression, '19' to '23' is visually lossless with minimal compression artifact, where 23 is lower quality, but smaller file size)

# Save new output folder file structure:
# drive > recording date > mouse ID > video directory (e.g. 'videos') > CameraN

# In the command window, you should see campy open N cameras (set by maxCams), load parameters (feature file), report the serial numbers, and await triggered frames. 
-An empty mp4 file (named 'fname.mp4') will be generated for each camera in the output 'CameraX' folders.
-When triggers start, campy's framegrabber function will "grab" frames and place them in each camera's buffer. 
-StartPipe compresses these buffered frames in order of acquisition and appends the frames to the output mp4 file.
-Once the desired number of frames are acquired (set by 'recTimeInSec'), or the user presses Ctrl+C, acquisition stops, the cameras close, and mp4 files save in the background. 
-Finally, a numpy file is saved, containing frame numbers and frame timestamps in each camera's output folder.

# Usage: in command line or anaconda prompt, run:
cd "basedir"
python campy.py
# Start hardware triggers



