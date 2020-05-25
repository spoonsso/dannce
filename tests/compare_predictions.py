"""
Compares two prediction matfiles
"""
import numpy as np
import scipy.io as sio
import sys

m1 = sio.loadmat(sys.argv[1])
m2 = sio.loadmat(sys.argv[2])

if 'com' in m1.keys():
    print("Checking for parity between COM predictions...")
    assert np.sum(m1['com']-m2['com']) == 0
    print("Good!")
elif 'pred' in m2.keys():
    print("Checking for parity between COM predictions...")
    assert np.sum(m1['pred']-m2['pred']) == 0
    print("Good!")
else:
    raise Exception("Expected fields (pred, com) not found in inputs")