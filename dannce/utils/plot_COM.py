"""
Use this script to make a plot of a pickle COM file's contents

Usage:
python plot_COM.py path_to_COM
"""

import sys
import numpy as np
import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from six.moves import cPickle

ppth = sys.argv[1]
pdir = os.path.dirname(ppth)

with open(ppth, 'rb') as f:
    com = cPickle.load(f)

keys = list(com.keys())
cams = list(com[keys[0]].keys())[1:]  # First key is not a camname
ncams = len(cams)

cm = np.zeros((ncams, len(keys), 2))

for (i, key) in enumerate(com.keys()):
    for (j, cam) in enumerate(cams):
        cm[j, i, :] = com[key][cam]['COM']

# Plot x- COM
f = plt.figure()
ax = f.add_subplot(111)
for i in range(ncams):
    ax.plot(cm[i, :, 0], label=cams[i])
ax.legend(fancybox=True).get_frame().set_alpha(0.5)
plt.savefig(os.path.join(pdir, 'COMx.png'))

# Plot y- COM
f = plt.figure()
ax = f.add_subplot(111)
for i in range(ncams):
    ax.plot(cm[i, :, 1], label=cams[i])
ax.legend(fancybox=True).get_frame().set_alpha(0.5)
plt.savefig(os.path.join(pdir, 'COMy.png'))