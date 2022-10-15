"""
Example script for visualizing Rat 7M data. 3D ground truth is projected into each
camera view, plotted into a figure, and saved as a png.

To load the correct video file, the input mocap data structure filename must contain the subject
number and recording day number as 'mocap-s{subject#}-d{day#}.mat'

Usage:
    python plot2DProjection.py [path_to_mocap_file (str)] [path_to_video_directory (str)] [sample number (int)]
"""

import imageio
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import dannce.engine.ops as ops
from dannce.utils.rat7m.loadStructs import load_mocap, load_cameras

_CHUNKS = 3500

mcpf = sys.argv[1]
vidf = sys.argv[2]
sID = int(sys.argv[3])

# Get subject ID and day ID from mocap file name
mcp_base = os.path.basename(os.path.normpath(mcpf))
s = mcp_base.split('s')[1][0]
d = mcp_base.split('d')[1][0]

# Load in mocap struct
mocap = load_mocap(mcpf)
cameras = load_cameras(mcpf)

# Loop over all cameras/vidoes. Load in the desired frame, p
#   roject the 3D points, and plot

fig, axs = plt.subplots(2, 3, figsize=(12, 8))

camnames = list(cameras.keys())
camnames = sorted(camnames, key=lambda x: int(x[-1]))
for i, cam in enumerate(camnames):

    # Load frame ----
    # Due to the way cameras are synchronized, the actual frame index for
    # each camera could be different. We use the cameras.frame variable
    # to grab this.

    fID = np.squeeze(cameras[cam]["frame"])[sID]
    vidID = fID // _CHUNKS * _CHUNKS
    vidIDf = fID % _CHUNKS

    vid_filename = "s{}-d{}-{}-{}.mp4".format(s, d, cam.lower(), vidID)
    vid_filename = os.path.join(vidf, vid_filename)
    print("----Frame {} from {}----".format(fID, cam))
    print("Reading frame {} from {}".format(vidIDf, vid_filename))
    vid = imageio.get_reader(vid_filename)
    im = vid.get_data(vidIDf)

    # Project 3D points into this camera using calibration parameters
    pts = mocap[sID].T
    pts = ops.project_to2d(pts,
                           cameras[cam]["IntrinsicMatrix"],
                           cameras[cam]["rotationMatrix"],
                           cameras[cam]["translationVector"])[:, :2]

    pts = ops.distortPoints(pts,
                            np.squeeze(cameras[cam]["IntrinsicMatrix"]),
                            np.squeeze(cameras[cam]["RadialDistortion"]),
                            np.squeeze(cameras[cam]["TangentialDistortion"])).T

    ax = axs[i // 3, i % 3]
    ax.imshow(im)
    ax.plot(pts[:, 0], pts[:, 1], '.r', markersize=5)
    ax.set_title(cam)
    ax.axis("off")

outname = "s{}-d{}-sample{}.png".format(s, d, sID)
print("Saving plot to " + outname)
plt.savefig(outname)
