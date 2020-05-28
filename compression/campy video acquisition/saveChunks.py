
import os
import imageio
import math
from subprocess import Popen
import sys
import time
import multiprocessing as mp

recDate = "20191030"
expt = "mouse11"
chunkLengthInFrames = 3000
numCams = 6

# Get metadata from video (assuming all cameras are the same)
basedir = os.path.join('D:\\',recDate,expt,'raw','Camera')
os.chdir(basedir + '1')
vid = imageio.get_reader('1.mp4')
fps = vid.get_meta_data()['fps']
durationInSec = vid.get_meta_data()['duration']
durationInFrames = fps*durationInSec
chunkLengthInSec = chunkLengthInFrames/fps
numChunks = math.floor(durationInFrames/chunkLengthInFrames)

def chunkFiles(camNum):
    os.chdir(basedir + str(camNum+1))

    startFrame = 0
    startTimeInSec = 0
    timeInSec = 0
    hrsStart = 0; minStart = 0; secStart = 0; msStart = 0

    for t in range(0,numChunks):
        #no need to pad zeros
        startTime = str(hrsStart) + ':' + str(minStart) + ':' + str(secStart) + '.' + str(msStart)

        timeEnd = startTimeInSec + chunkLengthInSec
        hr = math.floor(timeEnd/3600)
        timeEnd = timeEnd - hr*3600
        mn = math.floor(timeEnd/60)
        timeEnd = timeEnd - mn*60
        sc = math.floor(timeEnd)
        timeEnd = timeEnd - sc
        ms = math.floor(timeEnd*1000)

        endTime = str(hr) + ':' + str(mn) + ':' + str(sc) + '.' + str(ms)
        cmd = ('ffmpeg -y -i 1.mp4 -ss ' + startTime + ' -to ' + endTime + 
        ' -c:v copy -c:a copy ' + str(startFrame) + '.mp4' + ' -async 1 '
        ' -hide_banner -loglevel panic')

        p = Popen(cmd.split())

        startFrame = startFrame + chunkLengthInFrames
        startTimeInSec = startTimeInSec + chunkLengthInSec
        hrsStart = hr
        minStart = mn
        secStart = sc
        msStart = ms

        print('Copying video ' + str(camNum+1) + ' chunk ' + str(t) + '...')

if __name__ == '__main__':            
    
    ts = time.time()
    print('Chunking videos...')
    pp = mp.Pool(numCams)
    pp.map(chunkFiles,range(0,numCams))