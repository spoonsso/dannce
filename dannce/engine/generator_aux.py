"""Generator for 3d video images."""
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import os
import imageio
from dannce.engine import processing as processing
import scipy.io as sio
import warnings
import time
import matplotlib.pyplot as plt
from dannce.engine.video import LoadVideoFrame
from typing import Text, Tuple, List, Union, Dict
from lilab.cameras_setup import get_view_xywh_wrapper
from dannce.engine import ops as ops
from nvjpeg import NvJpeg

nj = NvJpeg()

class GeneratorAux(keras.utils.Sequence):
    def __init__(
        self,
        list_IDs: List,
        labels: Dict,
        labels_3d: Dict,
        camera_params: Dict,
        com3d: Dict,
        tifdirs: List,
        batch_size: int = 32,
        dim_in: Tuple = (32, 32, 32),
        n_channels_in: int = 1,
        n_channels_out: int = 1,
        shuffle: bool = True,
        camnames: List = [],
        vmin: int = -100,
        vmax: int = 100,
        nvox: int = 64,
        out_scale = 5,
        gpu_id: Text = "0",
        interp: Text = "linear",
        channel_combo=None,
        mode: Text = "3dprob",
        rotation: bool = False,
        vidreaders: Dict = None,
        distort: bool = True,
        expval: bool = False,
        multicam: bool = True,
        var_reg: bool = False,
        norm_im: bool = True,
        chunks: int = 3500,
        mono: bool = False,
        predict_flag: bool = False,
        exp_voxel_size = None,
        *args,
        **kargs
    ):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        if isinstance(nvox, int):
            nvox = (nvox, nvox, nvox)
        self.nvox = np.array(nvox)
        assert len(self.nvox)==3, "nvox must be length 3"
        assert self.nvox[0]==self.nvox[1], "nvox must be square"
        self.dim_out_3d = self.nvox  # length 3
        self.labels_3d = labels_3d
        self.camera_params = camera_params
        self.interp = interp
        self.channel_combo = channel_combo
        print(self.channel_combo)
        self.mode = mode
        self.com3d = com3d
        self.out_scale = out_scale
        self.rotation = rotation
        self.distort = distort
        self.expval = expval
        self.var_reg = var_reg
        # If saving npy as uint8 rather than training directly, dont normalize
        self.norm_im = norm_im
        self.gpu_id = gpu_id
        self.exp_voxel_size = exp_voxel_size
        self.COM_aug = 10

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def __data_generation(self, list_IDs_temp: List) -> Tuple:
        # Initialization
        first_exp = int(self.list_IDs[0].split("_")[0])

        X = np.zeros((self.batch_size, *self.dim_out_3d, len(self.camnames[first_exp])),dtype="float32")

        assert self.mode == "3dprob", "Only 3dprob heatmap mode supported"
        assert not self.expval
        assert self.mono
        y_3d = np.zeros((self.batch_size, *self.dim_out_3d, self.n_channels_out), dtype="float32")

        # Generate data
        for ibatch, ID in enumerate(list_IDs_temp):
            experimentID = int(ID.split("_")[0])

            # Voxel size refine
            if self.exp_voxel_size:
                voxel_size = self.exp_voxel_size[experimentID]
                vmin, vmax = -voxel_size / 2, voxel_size / 2
            else:
                vmin, vmax = self.vmin, self.vmax

            # For 3D ground truth
            this_y_3d = self.labels_3d[ID]
            this_COM_3d = self.com3d[ID]

            if self.COM_aug is not None:
                this_COM_3d = this_COM_3d.copy().ravel()
                this_COM_3d = (
                    this_COM_3d
                    + self.COM_aug * 2 * np.random.rand(len(this_COM_3d))
                    - self.COM_aug
                )
                this_COM_3d[2] += np.random.rand() * 40 - 20


            # Create and project the grid here,
            xticks = np.linspace(vmin, vmax, self.dim_out_3d[0])
            yticks = np.linspace(vmin, vmax, self.dim_out_3d[1])
            zticks = np.linspace(vmin, vmax, self.dim_out_3d[2])

            # Random flip the grid
            mirror_aug = np.random.rand() > 0.5
            if mirror_aug:
                xticks = xticks[::-1]
                flipidx = [0,2,1,3,4,5,8,9,6,7,12,13,10,11]
                this_y_3d = this_y_3d[:, flipidx]

            (x_box_3d, y_box_3d, z_box_3d) = np.meshgrid(xticks, yticks, zticks)

            # Random rotate the grid
            theta = np.random.rand() * 2 * np.pi
            x_box_3d_r = x_box_3d * np.cos(theta) - y_box_3d * np.sin(theta)
            y_box_3d_r = x_box_3d * np.sin(theta) + y_box_3d * np.cos(theta)
            z_box_3d_r = z_box_3d
            x_coord_3d = x_box_3d_r + this_COM_3d[0]
            y_coord_3d = y_box_3d_r + this_COM_3d[1]
            z_coord_3d = z_box_3d_r + this_COM_3d[2]

            assert self.n_channels_out == this_y_3d.shape[1]
            for j in range(self.n_channels_out):
                y_3d[ibatch, ..., j] = np.exp(
                    -(
                        (x_coord_3d - this_y_3d[0, j]) ** 2
                        + (y_coord_3d - this_y_3d[1, j]) ** 2
                        + (z_coord_3d - this_y_3d[2, j]) ** 2
                    )
                    / (2 * self.out_scale ** 2)
                )

            views = get_view_xywh_wrapper(len(self.camera_params[experimentID]))
            imfile = self.labels[ID]
            im_canvas = nj.read(imfile)
            ims = [im_canvas[y:y+h, x:x+w] for (x,y,w,h) in views]

            for _ci, camname in enumerate(self.camnames[experimentID]):
                thisim = ims[_ci]
                # Project de novo or load in approximate (faster)
                # TODO(break up): This is hard to read, consider breaking up
                proj_grid = ops.project_to2d(
                    np.stack(
                        (
                            x_coord_3d.ravel(),
                            y_coord_3d.ravel(),
                            z_coord_3d.ravel(),
                        ),
                        axis=1,
                    ),
                    self.camera_params[experimentID][camname]["K"],
                    self.camera_params[experimentID][camname]["R"],
                    self.camera_params[experimentID][camname]["t"],
                )

                assert self.distort
                proj_grid = ops.distortPoints(
                    proj_grid[:, :2],
                    self.camera_params[experimentID][camname]["K"],
                    np.squeeze(
                        self.camera_params[experimentID][camname]["RDistort"]
                    ),
                    np.squeeze(
                        self.camera_params[experimentID][camname]["TDistort"]
                    ),
                ).T

                (r, g, b) = ops.sample_grid(thisim, proj_grid, method=self.interp)

                X[ibatch, :, :, :, _ci] = np.reshape(
                    r, (self.nvox, self.nvox, self.nvox)
                )

        assert not self.channel_combo=="avg"

        if self.channel_combo == "random":
            X = X[:, :, :, :, :, np.random.permutation(X.shape[-1])]

        if self.norm_im:
            return processing.preprocess_3d(X), y_3d
        else:
            return X, y_3d