"""
Reads in COM3D pickle file, takes median across camera pairs, and saves the result
    into a target Label3D file for access w/ dannce.

    Usage: python COMmat_to_Label3D.py [path_to_com.mat_file] [path_to_label3d_file]
"""
import numpy as np
import sys
import scipy.io as sio
from six.moves import cPickle
import os

ifile = sio.loadmat(sys.argv[1])
ofile = sio.loadmat(sys.argv[2])

# save temp backup, will be deleted later
sio.savemat(sys.argv[2]+".temp", ofile)

com = {}
com["com3d"] = ifile["com"]
com["sampleID"] = ifile["sampleID"]

ofile["com"] = com

sio.savemat(sys.argv[2], ofile)

print("Removing temp file...")
os.remove(sys.argv[2]+".temp")
print("Removed temp file.")