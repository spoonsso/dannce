"""Generator module for dannce training.
"""
import os
import numpy as np
from tensorflow import keras
from dannce.engine import processing_cxf as processing
from dannce.engine import ops as ops
from dannce.engine.video import LoadVideoFrame
import imageio
from nvjpeg import NvJpeg
import warnings
import time
import scipy.ndimage.interpolation
import tensorflow as tf
from typing import Union
from lilab.cameras_setup import get_view_xywh_wrapper
from dannce.engine.com_detection_cxf import ims_to_com2ds, matlab_pose_to_cv2_pose, com2ds_to_com3d
import torch

# from tensorflow_graphics.geometry.transformation.axis_angle import rotate
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
from typing import List, Dict, Tuple, Text
import ffmpegcv
from lilab.mmdet_dev.canvas_reader import CanvasReader
from lilab.cvutils_new.canvas_reader_pannel import CanvasReaderPannelMask, CanvasReaderPannel
nj = NvJpeg()

class DataGenerator(keras.utils.Sequence):
    """Generate data for Keras.

    Attributes:
        batch_size (int): Batch size to generate
        camnames (List): List of camera names.
        clusterIDs (List): List of sampleIDs
        crop_height (Tuple): (first, last) pixels in image height
        crop_width (tuple): (first, last) pixels in image width
        currvideo (Dict): Contains open video objects
        currvideo_name (Dict): Contains open video object names
        dim_in (Tuple): Input dimension
        dim_out (Tuple): Output dimension
        extension (Text): Video extension
        indexes (np.ndarray): sample indices used for batch generation
        labels (Dict): Label dictionary
        list_IDs (List): List of sampleIDs
        mono (bool): If True, use grayscale image.
        n_channels_in (int): Number of input channels
        n_channels_out (int): Number of output channels
        out_scale (int): Scale of the output gaussians.
        samples_per_cluster (int): Samples per cluster
        shuffle (bool): If True, shuffle the samples.
        vidreaders (Dict): Dict containing video readers.
        predict_flag (bool): If True, use imageio for reading videos, rather than OpenCV
    """

    def __init__(
        self,
        list_IDs: List,
        labels: Dict,
        clusterIDs: List,
        batch_size: int = 32,
        dim_in: Tuple = (32, 32, 32),
        n_channels_in: int = 1,
        n_channels_out: int = 1,
        out_scale: float = 5,
        shuffle: bool = True,
        camnames: List = [],
        crop_width: Tuple = (0, 1024),
        crop_height: Tuple = (20, 1300),
        samples_per_cluster: int = 0,
        vidreaders: Dict = None,
        chunks: int = 3500,
        mono: bool = False,
        mirror: bool = False,
        predict_flag: bool = False,
    ):
        """Initialize Generator.

        Args:
            list_IDs (List): List of sampleIDs
            labels (Dict): Label dictionary
            clusterIDs (List): List of sampleIDs
            batch_size (int, optional): Batch size to generate
            dim_in (Tuple, optional): Input dimension
            n_channels_in (int, optional): Number of input channels
            n_channels_out (int, optional): Number of output channels
            out_scale (float, optional): Scale of the output gaussians.
            shuffle (bool, optional): If True, shuffle the samples.
            camnames (List, optional): List of camera names.
            crop_width (Tuple, optional): (first, last) pixels in image width
            crop_height (Tuple, optional): (first, last) pixels in image height
            samples_per_cluster (int, optional): Samples per cluster
            vidreaders (Dict, optional): Dict containing video readers.
            chunks (int, optional): Size of chunks when using chunked mp4.
            mono (bool, optional): If True, use grayscale image.
            predict_flag (bool, optional): If True, use imageio for reading videos, rather than OpenCV
        """
        self.dim_in = dim_in
        self.dim_out = dim_in
        self.batch_size = batch_size
        self.labels = labels
        self.vidreaders = vidreaders
        self.list_IDs = list_IDs
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.shuffle = shuffle
        # sigma for the ground truth joint probability map Gaussians
        self.out_scale = out_scale
        self.camnames = camnames
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.clusterIDs = clusterIDs
        self.samples_per_cluster = samples_per_cluster
        self._N_VIDEO_FRAMES = chunks
        self.mono = mono
        self.mirror = mirror
        self.predict_flag = predict_flag
        self.on_epoch_end()

        if self.vidreaders is not None:
            self.extension = (
                "." + list(vidreaders[camnames[0][0]].keys())[0].rsplit(".")[-1]
            )

        assert len(self.list_IDs) == len(self.clusterIDs)

        self.load_frame = None

    def __len__(self) -> int:
        """Denote the number of batches per epoch.

        Returns:
            int: Batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        """Update indexes after each epoch."""
        self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def random_rotate(self, X: np.ndarray, y_3d: np.ndarray, log: bool = False):
        """Rotate each sample by 0, 90, 180, or 270 degrees.

        log indicates whether to return the rotation pattern (for saving) as well.

        Args:
            X (np.ndarray): Input images
            y_3d (np.ndarray): Output 3d targets
            log (bool, optional): If True, log the rotations.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Rotated X and y_3d.

            or

            Tuple[np.ndarray, np.ndarray, np.ndarray]: Rotated X, y_3d, and rot val
        """
        rots = np.random.choice(np.arange(4), X.shape[0])
        for i in range(X.shape[0]):
            if rots[i] == 0:
                pass
            elif rots[i] == 1:
                # Rotate180
                X[i] = self.rot180(X[i])
                y_3d[i] = self.rot180(y_3d[i])
            elif rots[i] == 2:
                # Rotate90
                X[i] = self.rot90(X[i])
                y_3d[i] = self.rot90(y_3d[i])
            elif rots[i] == 3:
                # Rotate -90/270
                X[i] = self.rot90(X[i])
                X[i] = self.rot180(X[i])
                y_3d[i] = self.rot90(y_3d[i])
                y_3d[i] = self.rot180(y_3d[i])

        if log:
            return X, y_3d, rots
        else:
            return X, y_3d


class DataGenerator_3Dconv(DataGenerator):
    """Update generator class to handle multiple experiments.

    Attributes:
        camera_params (Dict): Camera parameters dictionary.
        channel_combo (Text): Method for shuffling camera input order
        com3d (Dict): Dictionary of com3d data.
        COM_aug (bool): If True, augment the COM.
        crop_im (bool): If True, crop images.
        depth (bool): If True, appends voxel depth to sampled image features [DEPRECATED]
        dim_out_3d (Tuple): Dimensions of the 3D volume, in voxels
        distort (bool): If true, apply camera undistortion.
        expval (bool): If True, process an expected value network (AVG)
        gpu_id (Text): Identity of GPU to use.
        immode (Text): Toggles using 'video' or 'tif' files as image input [DEPRECATED]
        interp (Text): Interpolation method.
        labels_3d (Dict): Contains ground-truth 3D label coordinates.
        mode (Text): Toggles output label format to match MAX vs. AVG network requirements.
        multicam (bool): If True, formats data to work with multiple cameras as input.
        norm_im (bool): If True, normalize images.
        nvox (int): Number of voxels per box side
        rotation (bool): If True, use simple rotation augmentation.
        tifdirs (List): Directories of .tifs
        var_reg (bool): If True, adds a variance regularization term to the loss function.
        vmax (int): Maximum box dim (relative to the COM)
        vmin (int): Minimum box dim (relative to the COM)
        vsize (float): Side length of one voxel
        predict_flag (bool): If True, use imageio for reading videos, rather than OpenCV
    """

    def __init__(
        self,
        list_IDs: List,
        labels: Dict,
        labels_3d: Dict,
        camera_params: Dict,
        clusterIDs: List,
        com3d: Dict,
        tifdirs: List,
        batch_size: int = 32,
        dim_in: Tuple = (32, 32, 32),
        n_channels_in: int = 1,
        n_channels_out: int = 1,
        out_scale: int = 5,
        shuffle: bool = True,
        camnames: List = [],
        crop_width: Tuple = (0, 1024),
        crop_height: Tuple = (20, 1300),
        vmin: int = -100,
        vmax: int = 100,
        nvox: int = 32,
        gpu_id: Text = "0",
        interp: Text = "linear",
        depth: bool = False,
        channel_combo=None,
        mode: Text = "3dprob",
        samples_per_cluster: int = 0,
        immode: Text = "tif",
        rotation: bool = False,
        vidreaders: Dict = None,
        distort: bool = True,
        expval: bool = False,
        multicam: bool = True,
        var_reg: bool = False,
        COM_aug: bool = None,
        crop_im: bool = True,
        norm_im: bool = True,
        chunks: int = 3500,
        mono: bool = False,
        mirror: bool = False,
        predict_flag: bool = False,
        exp_voxel_size = None
    ):
        """Initialize data generator.

        Args:
            list_IDs (List): List of sample Ids
            labels (Dict): Dictionary of labels
            labels_3d (Dict): Dictionary of 3d labels.
            camera_params (Dict): Camera parameters dictionary.
            clusterIDs (List): List of sample Ids
            com3d (Dict): Dictionary of com3d data.
            tifdirs (List): Directories of .tifs
            batch_size (int, optional): Batch size to generate
            dim_in (Tuple, optional): Input dimension
            n_channels_in (int, optional): Number of input channels
            n_channels_out (int, optional): Number of output channels
            out_scale (int, optional): Scale of the output gaussians.
            shuffle (bool, optional): If True, shuffle the samples.
            camnames (List, optional): List of camera names.
            crop_width (Tuple, optional): (first, last) pixels in image width
            crop_height (Tuple, optional): (first, last) pixels in image height
            vmin (int, optional): Minimum box dim (relative to the COM)
            vmax (int, optional): Maximum box dim (relative to the COM)
            nvox (int, optional): Number of voxels per box side
            gpu_id (Text, optional): Identity of GPU to use.
            interp (Text, optional): Interpolation method.
            depth (bool): If True, appends voxel depth to sampled image features [DEPRECATED]
            channel_combo (Text): Method for shuffling camera input order
            mode (Text): Toggles output label format to match MAX vs. AVG network requirements.
            samples_per_cluster (int, optional): Samples per cluster
            immode (Text): Toggles using 'video' or 'tif' files as image input [DEPRECATED]
            rotation (bool, optional): If True, use simple rotation augmentation.
            vidreaders (Dict, optional): Dict containing video readers.
            distort (bool, optional): If true, apply camera undistortion.
            expval (bool, optional): If True, process an expected value network (AVG)
            multicam (bool): If True, formats data to work with multiple cameras as input.
            var_reg (bool): If True, adds a variance regularization term to the loss function.
            COM_aug (bool, optional): If True, augment the COM.
            crop_im (bool, optional): If True, crop images.
            norm_im (bool, optional): If True, normalize images.
            chunks (int, optional): Size of chunks when using chunked mp4.
            mono (bool, optional): If True, use grayscale image.
            predict_flag (bool, optional): If True, use imageio for reading videos, rather than OpenCV.
            exp_voxel_size (None or dict): If not None, use this voxel size for the expected value network.
        """
        DataGenerator.__init__(
            self,
            list_IDs,
            labels,
            clusterIDs,
            batch_size,
            dim_in,
            n_channels_in,
            n_channels_out,
            out_scale,
            shuffle,
            camnames,
            crop_width,
            crop_height,
            samples_per_cluster,
            vidreaders,
            chunks,
            mono,
            mirror,
            predict_flag,
        )
        self.vmin = vmin
        self.vmax = vmax
        self.nvox = nvox
        self.vsize = (vmax - vmin) / nvox
        self.dim_out_3d = (nvox, nvox, nvox)
        self.labels_3d = labels_3d
        self.camera_params = camera_params
        self.interp = interp
        self.depth = depth
        self.channel_combo = channel_combo
        print(self.channel_combo)
        self.mode = mode
        self.immode = immode
        self.tifdirs = tifdirs
        self.com3d = com3d
        self.rotation = rotation
        self.distort = distort
        self.expval = expval
        self.multicam = multicam
        self.var_reg = var_reg
        self.COM_aug = COM_aug
        self.crop_im = crop_im
        # If saving npy as uint8 rather than training directly, dont normalize
        self.norm_im = norm_im
        self.gpu_id = gpu_id
        self.exp_voxel_size = exp_voxel_size

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate one batch of data.

        Args:
            index (int): Frame index

        Returns:
            Tuple[np.ndarray, np.ndarray]: One batch of data
                X (np.ndarray): Input volume
                y (np.ndarray): Target
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def rot90(self, X: np.ndarray) -> np.ndarray:
        """Rotate X by 90 degrees CCW.

        Args:
            X (np.ndarray): Input volume.

        Returns:
            np.ndarray: Rotated volume
        """
        X = np.transpose(X, [1, 0, 2, 3])
        X = X[:, ::-1, :, :]
        return X

    def rot180(self, X):
        """Rotate X by 180 degrees.

        Args:
            X (np.ndarray): Input volume.

        Returns:
            np.ndarray: Rotated volume
        """
        X = X[::-1, ::-1, :, :]
        return X

    def __data_generation(self, list_IDs_temp: List) -> Tuple:
        """Generate data containing batch_size samples.

        X : (n_samples, *dim, n_channels)

        Args:
            list_IDs_temp (List): List of experiment Ids

        Returns:
            Tuple: Batch_size training samples
                X: Input volumes
                y_3d: Targets
                rotangle: Rotation angle
        Raises:
            Exception: Invalid generator mode specified.
        """
        # Initialization
        first_exp = int(self.list_IDs[0].split("_")[0])

        X = np.zeros(
            (
                self.batch_size * len(self.camnames[first_exp]),
                *self.dim_out_3d,
                self.n_channels_in + self.depth,
            ),
            dtype="float32",
        )

        if self.mode == "3dprob":
            y_3d = np.zeros(
                (self.batch_size, self.n_channels_out, *self.dim_out_3d),
                dtype="float32",
            )
        elif self.mode == "coordinates":
            y_3d = np.zeros((self.batch_size, 3, self.n_channels_out), dtype="float32")
        else:
            raise Exception("not a valid generator mode")

        if self.expval:
            sz = self.dim_out_3d[0] * self.dim_out_3d[1] * self.dim_out_3d[2]
            X_grid = np.zeros((self.batch_size, sz, 3), dtype="float32")

        # Generate data
        cnt = 0

        for i, ID in enumerate(list_IDs_temp):
            experimentID = int(ID.split("_")[0])

            # Voxel size refine
            if self.exp_voxel_size:
                voxel_size = self.exp_voxel_size[experimentID]
                vmin, vmax = -voxel_size / 2, voxel_size / 2
                vsize = voxel_size / self.nvox
            else:
                vmin, vmax = self.vmin, self.vmax
                vsize = self.vsize

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

            # Create and project the grid here,
            grids = np.arange(vmin + vsize/2, vmax, vsize)
            xgrid = grids + this_COM_3d[0]
            ygrid = grids + this_COM_3d[1]
            zgrid = grids + this_COM_3d[2]
            (x_coord_3d, y_coord_3d, z_coord_3d) = np.meshgrid(xgrid, ygrid, zgrid)

            if self.mode == "3dprob":
                for j in range(self.n_channels_out):
                    y_3d[i, j] = np.exp(
                        -(
                            (y_coord_3d - this_y_3d[1, j]) ** 2
                            + (x_coord_3d - this_y_3d[0, j]) ** 2
                            + (z_coord_3d - this_y_3d[2, j]) ** 2
                        )
                        / (2 * self.out_scale ** 2)
                    )
                    # When the voxel grid is coarse, we will likely miss
                    # the peak of the probability distribution, as it
                    # will lie somewhere in the middle of a large voxel.
                    # So here we renormalize to [~, 1]

            if self.mode == "coordinates":
                if this_y_3d.shape == y_3d[i].shape:
                    y_3d[i] = this_y_3d
                else:
                    msg = "Note: ignoring dimension mismatch in 3D labels"
                    warnings.warn(msg)

            if self.expval:
                X_grid[i] = np.stack(
                    (
                        x_coord_3d.ravel(),
                        y_coord_3d.ravel(),
                        z_coord_3d.ravel(),
                    ),
                    axis=1,
                )

            views = get_view_xywh_wrapper(len(self.camera_params[experimentID]))
            imfile = self.labels[ID]
            im_canvas = imageio.imread(imfile) #old
            # im_canvas = nj.read(imfile) #new
            ims = [im_canvas[y:y+h, x:x+w] for (x,y,w,h) in views]

            for _ci, camname in enumerate(self.camnames[experimentID]):
                ts = time.time()
                thisim = ims[_ci]

                if self.mirror:
                    # Save copy of the first image loaded in, so that it can be flipped accordingly.
                    self.raw_im = thisim.copy()

                if self.mirror and self.camera_params[experimentID][camname]["m"] == 1:
                    thisim = self.raw_im.copy()
                    thisim = thisim[-1::-1]
                elif self.mirror and self.camera_params[experimentID][camname]["m"] == 0:
                    thisim = self.raw_im
                elif self.mirror:
                    raise Exception("Invalid mirror parameter, m, must be 0 or 1")

                # Project de novo or load in approximate (faster)
                # TODO(break up): This is hard to read, consider breaking up
                ts = time.time()
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

                if self.depth:
                    d = proj_grid[:, 2]
                # print("2D Proj took {} sec".format(time.time() - ts))
                ts = time.time()
                if self.distort:
                    """
                    Distort points using lens distortion parameters
                    """
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
                # print("Distort took {} sec".format(time.time() - ts))


                (r, g, b) = ops.sample_grid(thisim, proj_grid, method=self.interp)
                # print("Sample grid took {} sec".format(time.time() - ts))

                X[cnt, :, :, :, 0] = np.reshape(
                        r, (self.nvox, self.nvox, self.nvox)
                )
                X[cnt, :, :, :, 1] = np.reshape(
                    g, (self.nvox, self.nvox, self.nvox)
                )
                X[cnt, :, :, :, 2] = np.reshape(
                    b, (self.nvox, self.nvox, self.nvox)
                )
                if self.depth:
                    X[cnt, :, :, :, 3] = np.reshape(
                        d, (self.nvox, self.nvox, self.nvox)
                    )

                cnt = cnt + 1
                # print("Projection grid took {} sec".format(time.time() - tss))

        if self.multicam:
            X = np.reshape(
                X,
                (
                    self.batch_size,
                    len(self.camnames[first_exp]),
                    X.shape[1],
                    X.shape[2],
                    X.shape[3],
                    X.shape[4],
                ),
            )
            X = np.transpose(X, [0, 2, 3, 4, 5, 1])

            if self.channel_combo == "avg":
                X = np.nanmean(X, axis=-1)
            # Randomly reorder the cameras fed into the first layer
            elif self.channel_combo == "random":
                X = X[:, :, :, :, :, np.random.permutation(X.shape[-1])]
                X = np.reshape(
                    X,
                    (
                        X.shape[0],
                        X.shape[1],
                        X.shape[2],
                        X.shape[3],
                        X.shape[4] * X.shape[5],
                    ),
                    order="F",
                )
            else:
                X = np.reshape(
                    X,
                    (
                        X.shape[0],
                        X.shape[1],
                        X.shape[2],
                        X.shape[3],
                        X.shape[4] * X.shape[5],
                    ),
                    order="F",
                )
        else:
            # Then leave the batch_size and num_cams combined
            y_3d = np.tile(y_3d, [len(self.camnames[experimentID]), 1, 1, 1, 1])

        if self.mode == "3dprob":
            y_3d = np.transpose(y_3d, [0, 2, 3, 4, 1])

        if self.rotation:
            if self.expval:
                # First make X_grid 3d
                X_grid = np.reshape(
                    X_grid,
                    (self.batch_size, self.nvox, self.nvox, self.nvox, 3),
                )

                X, X_grid = self.random_rotate(X, X_grid)
                # Need to reshape back to raveled version
                X_grid = np.reshape(X_grid, (self.batch_size, -1, 3))
            else:
                X, y_3d = self.random_rotate(X, y_3d)

        if self.mono and self.n_channels_in == 3:
            # Convert from RGB to mono using the skimage formula. Drop the duplicated frames.
            # Reshape so RGB can be processed easily.
            X = np.reshape(
                X,
                (
                    X.shape[0],
                    X.shape[1],
                    X.shape[2],
                    X.shape[3],
                    self.n_channels_in,
                    -1,
                ),
                order="F",
            )
            X = (
                X[:, :, :, :, 0] * 0.2125
                + X[:, :, :, :, 1] * 0.7154
                + X[:, :, :, :, 2] * 0.0721
            )

        # Then we also need to return the 3d grid center coordinates,
        # for calculating a spatial expected value
        # Xgrid is typically symmetric for 90 and 180 degree rotations
        # (when vmax and vmin are symmetric)
        # around the z-axis, so no need to rotate X_grid.
        if self.expval:
            if self.var_reg:
                return (
                    [processing.preprocess_3d(X), X_grid],
                    [y_3d, np.zeros((self.batch_size, 1))],
                )

            if self.norm_im:
                # y_3d is in coordinates here.
                return [processing.preprocess_3d(X), X_grid], y_3d
            else:
                return [X, X_grid], y_3d
        else:
            if self.norm_im:
                return processing.preprocess_3d(X), y_3d
            else:
                return X, y_3d


class DataGenerator_3Dconv_torch_aug(DataGenerator):
    def __init__(
        self,
        list_IDs,
        labels,
        labels_3d,
        camera_params,
        clusterIDs,
        com3d,
        tifdirs,
        batch_size=32,
        dim_in=(32, 32, 32),
        n_channels_in=1,
        n_channels_out=1,
        out_scale=5,
        shuffle=True,
        camnames=[],
        crop_width=(0, 1024),
        crop_height=(20, 1300),
        vmin=-100,
        vmax=100,
        nvox=32,
        gpu_id="0",
        interp="linear",
        depth=False,
        channel_combo=None,
        mode="3dprob",
        samples_per_cluster=0,
        immode="tif",
        rotation=False,
        vidreaders=None,
        distort=True,
        expval=False,
        multicam=True,
        var_reg=False,
        COM_aug=None,
        crop_im=True,
        norm_im=True,
        chunks=3500,
        mono=False,
        mirror=False,
        predict_flag=False,
        exp_voxel_size = None
    ):
        DataGenerator.__init__(
            self,
            list_IDs,
            labels,
            clusterIDs,
            batch_size,
            dim_in,
            n_channels_in,
            n_channels_out,
            out_scale,
            shuffle,
            camnames,
            crop_width,
            crop_height,
            samples_per_cluster,
            vidreaders,
            chunks,
            mono,
            mirror,
            predict_flag,
        )
        self.vmin = vmin
        self.vmax = vmax
        self.nvox = nvox
        self.vsize = (vmax - vmin) / nvox
        self.dim_out_3d = (nvox, nvox, nvox)
        self.labels_3d = labels_3d
        self.camera_params = camera_params
        self.interp = interp
        self.depth = depth
        self.channel_combo = channel_combo
        print('Real voxel size: {}mm'.format(self.vmax - self.vmin))
        self.gpu_id = gpu_id
        self.mode = mode
        self.immode = immode
        self.tifdirs = tifdirs
        self.com3d = com3d
        self.rotation = rotation
        self.distort = distort
        self.expval = expval
        self.multicam = multicam
        self.var_reg = var_reg
        self.COM_aug = COM_aug
        self.crop_im = crop_im
        # If saving npy as uint8 rather than training directly, dont normalize
        self.norm_im = norm_im
        self.exp_voxel_size = exp_voxel_size

        # importing torch here allows other modes to run without pytorch installed
        gpu_id_tf = os.getenv('CUDA_VISIBLE_DEVICES').split(',')[int(gpu_id)]
        self.device = torch.device(f"cuda:{gpu_id_tf}")
        # self.device = torch.device('cpu')

        # self.threadpool = ThreadPool(len(self.camnames[0]))

        ts = time.time()
        # Limit GPU memory usage by Tensorflow to leave memory for PyTorch
        gpus = tf.config.list_physical_devices("GPU")
        # set visible device as the first GPU
        gpu_id_tf = os.getenv('CUDA_VISIBLE_DEVICES').split(',')[int(gpu_id)]
        # tf.config.experimental.set_visible_devices(gpus[int(gpu_id_tf)], "GPU")
        # # set memory growth to True
        # tf.config.experimental.set_memory_growth(gpus[int(gpu_id_tf)], True)

        tf.config.set_visible_devices(gpus[int(gpu_id_tf)], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[int(gpu_id_tf)], True)
        # config = tf.compat.v1.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.45
        # config.gpu_options.allow_growth = True
        # self.session = tf.compat.v1.InteractiveSession(config=config, graph=tf.Graph())

        experimentIDs = set([int(ID.split("_")[0]) for ID in list_IDs])

        for experimentID in experimentIDs:
            for camname in self.camnames[experimentID]:
                # M only needs to be computed once for each camera
                K = self.camera_params[experimentID][camname]["K"]
                R = self.camera_params[experimentID][camname]["R"]
                t = self.camera_params[experimentID][camname]["t"]
                M = torch.as_tensor(
                    ops.camera_matrix(K, R, t), dtype=torch.float32
                )
                self.camera_params[experimentID][camname]["M"] = M

        print("Init took {} sec.".format(time.time() - ts))

    def __getitem__(self, index: int):
        """Generate one batch of data.

        Args:
            index (int): Frame index

        Returns:
            Tuple[np.ndarray, np.ndarray]: One batch of data X
                (np.ndarray): Input volume y 
                (np.ndarray): Target
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def rot90(self, X):
        """Rotate X by 90 degrees CCW.

        Args:
            X (np.ndarray): Volume

        Returns:
            X (np.ndarray): Rotated volume
        """
        X = X.permute(1, 0, 2, 3)
        X = X.flip(1)
        return X

    def rot180(self, X):
        """Rotate X by 180 degrees.

        Args:
            X (np.ndarray): Volume

        Returns:
            X (np.ndarray): Rotated volume
        """
        X = X.flip(0).flip(1)
        return X

    def project_grid(self, X_grid, camname, ID, experimentID, thisim):
        """Projects 3D voxel centers and sample images as projected 2D pixel coordinates

        Args:
            X_grid (np.ndarray): 3-D array containing center coordinates of each voxel.
            camname (Text): camera name
            ID (Text): string denoting a sample ID
            experimentID (int): identifier for a video recording session.

        Returns:
            np.ndarray: projected voxel centers, now in 2D pixels
        """
        ts = time.time()
        # Need this copy so that this_y does not change
        this_y = torch.as_tensor(
            torch.zeros(2,2) * torch.nan,
            dtype=torch.float32,
            device=self.device,
        ).round()

        if torch.all(torch.isnan(this_y)):
            com_precrop = torch.zeros_like(this_y[:, 0]) * torch.nan
        else:
            # For projecting points, we should not use this offset
            com_precrop = torch.mean(this_y, axis=1)

        this_y[0, :] = this_y[0, :] - self.crop_width[0]
        this_y[1, :] = this_y[1, :] - self.crop_height[0]
        com = torch.mean(this_y, axis=1)

        return self.pj_grid_post(X_grid, camname, ID, experimentID,
                                 com, com_precrop, thisim)

    def pj_grid_mirror(self, X_grid, camname, ID, experimentID, thisim):
        this_y = torch.as_tensor(
            [torch.nan, torch.nan, torch.nan],
            dtype=torch.float32,
            device=self.device,
        ).round()

        if torch.all(torch.isnan(this_y)):
            com_precrop = torch.zeros_like(this_y[:, 0]) * torch.nan
        else:
            # For projecting points, we should not use this offset
            com_precrop = torch.mean(this_y, axis=1)

        this_y[0, :] = this_y[0, :] - self.crop_width[0]
        this_y[1, :] = this_y[1, :] - self.crop_height[0]
        com = torch.mean(this_y, axis=1)

        if not self.mirror:
            raise Exception("Trying to project onto mirrored images without mirror being set properly")

        if self.camera_params[experimentID][camname]["m"] == 1:
            passim = thisim[-1::-1].copy()
        elif self.camera_params[experimentID][camname]["m"] == 0:
            passim = thisim.copy()
        else:
            raise Exception("Invalid mirror parameter, m, must be 0 or 1")


        return self.pj_grid_post(X_grid, camname, ID, experimentID,
                                 com, com_precrop, passim)

    def pj_grid_post(self, X_grid, camname, ID, experimentID,
                     com, com_precrop, thisim):
        # separate the porjection and sampling into its own function so that
        # when mirror == True, this can be called directly
        ts = time.time()
        if self.crop_im:
            if torch.all(torch.isnan(com)):
                thisim = torch.zeros(
                    (self.dim_in[1], self.dim_in[0], self.n_channels_in),
                    dtype=torch.uint8,
                    device=self.device,
                )
            else:
                thisim = processing.cropcom(thisim, com, size=self.dim_in[0])
        # print('Frame loading took {} sec.'.format(time.time() - ts))

        # ts = time.time()
        proj_grid = ops.project_to2d_torch(
            X_grid, self.camera_params[experimentID][camname]["M"], self.device
        )
        # print('Project2d took {} sec.'.format(time.time() - ts))

        # ts = time.time()
        if self.distort:
            proj_grid = ops.distortPoints_torch(
                proj_grid[:, :2],
                self.camera_params[experimentID][camname]["K"],
                np.squeeze(self.camera_params[experimentID][camname]["RDistort"]),
                np.squeeze(self.camera_params[experimentID][camname]["TDistort"]),
                self.device,
            )
            proj_grid = proj_grid.transpose(0, 1)
            # print('Distort took {} sec.'.format(time.time() - ts))

        # ts = time.time()
        if self.crop_im:
            proj_grid = proj_grid[:, :2] - com_precrop + self.dim_in[0] // 2
            # Now all coordinates should map properly to the image cropped around the COM
        else:
            # Then the only thing we need to correct for is crops at the borders
            proj_grid = proj_grid[:, :2]
            proj_grid[:, 0] = proj_grid[:, 0] - self.crop_width[0]
            proj_grid[:, 1] = proj_grid[:, 1] - self.crop_height[0]

        rgb = ops.sample_grid_torch(thisim, proj_grid, self.device, method=self.interp)
        # print('Sample grid {} sec.'.format(time.time() - ts))

        if (
            ~torch.any(torch.isnan(com_precrop))
            or (self.channel_combo == "avg")
            or not self.crop_im
        ):
            X = rgb.permute(0, 2, 3, 4, 1)

        return X

    # TODO(nesting): There is pretty deep locigal nesting in this function,
    # might be useful to break apart
    def __data_generation(self, list_IDs_temp):
        """Generate data containing batch_size samples.
        X : (n_samples, *dim, n_channels)

        Args:
            list_IDs_temp (List): List of experiment Ids

        Returns:
            Tuple: Batch_size training samples
                X: Input volumes
                y_3d: Targets
                rotangle: Rotation angle
        Raises:
            Exception: Invalid generator mode specified.
        """
        # Initialization
        first_exp = int(self.list_IDs[0].split("_")[0])

        X = torch.zeros(
            (
                self.batch_size * len(self.camnames[first_exp]),
                *self.dim_out_3d,
                self.n_channels_in + self.depth,
            ),
            dtype=torch.uint8,
            device=self.device,
        )

        if self.mode == "3dprob":
            y_3d = torch.zeros(
                (self.batch_size, self.n_channels_out, *self.dim_out_3d),
                dtype=torch.float32,
                device=self.device,
            )
        elif self.mode == "coordinates":
            y_3d = torch.zeros(
                (self.batch_size, 3, self.n_channels_out),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            raise Exception("not a valid generator mode")

        sz = self.dim_out_3d[0] * self.dim_out_3d[1] * self.dim_out_3d[2]
        X_grid = torch.zeros(
            (self.batch_size, sz, 3),
            dtype=torch.float32,
            device=self.device,
        )


        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            ts = time.time()
            sampleID = int(ID.split("_")[1])
            experimentID = int(ID.split("_")[0])

            # Voxel size refine
            if self.exp_voxel_size:
                voxel_size = self.exp_voxel_size[experimentID]
                vmin, vmax = -voxel_size / 2, voxel_size / 2
                vsize = voxel_size / self.nvox
            else:
                vmin, vmax = self.vmin, self.vmax
                vsize = self.vsize

            # For 3D ground truth
            this_y_3d = torch.as_tensor(
                self.labels_3d[ID],
                dtype=torch.float32,
                device=self.device,
            )
            this_COM_3d = torch.as_tensor(
                self.com3d[ID], dtype=torch.float32, device=self.device
            )
            this_COM_3d[2] += torch.rand([], device=self.device) * 40 - 20  # random z-axis offset
            # Create and project the grid here,
            grids = torch.arange(vmin + vsize/2, vmax, vsize, 
                dtype=torch.float32,
                device=self.device)
            xgrid = grids + this_COM_3d[0]
            ygrid = grids + this_COM_3d[1]
            zgrid = grids + this_COM_3d[2]
            (x_coord_3d, y_coord_3d, z_coord_3d) = torch.meshgrid(
                xgrid, ygrid, zgrid
            )
            X_grid[i] = torch.stack(
                (
                    x_coord_3d.transpose(0, 1).flatten(),
                    y_coord_3d.transpose(0, 1).flatten(),
                    z_coord_3d.transpose(0, 1).flatten(),
                ),
                axis=1,
            )

            if self.mode == "3dprob":
                X_grid_i = X_grid[i].reshape(*x_coord_3d.shape, 3)
                for j in range(self.n_channels_out):
                    y_3d[i, j] = torch.exp(
                        - torch.sum((X_grid_i - this_y_3d[:,j])**2, dim=-1)
                        / (2 * self.out_scale ** 2)
                    )

            if self.mode == "coordinates":
                if this_y_3d.shape == y_3d[i].shape:
                    y_3d[i] = this_y_3d
                else:
                    msg = "Note: ignoring dimension mismatch in 3D labels"
                    warnings.warn(msg)

            # Compute projected images in parallel using multithreading
            ts = time.time()
            num_cams = len(self.camnames[experimentID])
            arglist = []
            if self.mirror:
                # Here we only load the video once, and then parallelize the projection
                # and sampling after mirror flipping. For setups that collect views
                # in a single imgae with the use of mirrors
                loadim = self.load_frame.load_vid_frame(
                    self.labels[ID]["frames"][self.camnames[experimentID][0]],
                    self.camnames[experimentID][0],
                    extension=self.extension,
                )[
                    self.crop_height[0]: self.crop_height[1],
                    self.crop_width[0]: self.crop_width[1],
                ]
                for c in range(num_cams):
                    arglist.append(
                        [X_grid[i], self.camnames[experimentID][c], ID, experimentID, loadim]
                    )
                # result = self.threadpool.starmap(self.pj_grid_mirror, arglist)
                result = [self.pj_grid_mirror(*arg) for arg in arglist]
            else:
                views = get_view_xywh_wrapper(len(self.camera_params[experimentID]))
                imfile = self.labels[ID]
                im_canvas = imageio.imread(imfile)  #old
                # im_canvas = nj.read(imfile)  #new
                # print('Load image tooks {} sec.'.format(time.time()-ts))
                ims = [im_canvas[y:y+h, x:x+w] for (x,y,w,h) in views]
                for c in range(num_cams):
                    arglist.append(
                        [X_grid[i], self.camnames[experimentID][c], ID, experimentID, ims[c]]
                    )
                # result = self.threadpool.starmap(self.project_grid, arglist)
                ts = time.time()
                result = [self.project_grid(*arg) for arg in arglist]
                # print('Project took {} sec.'.format(time.time()-ts))

            for c in range(num_cams):
                ic = c + i * len(self.camnames[experimentID])
                X[ic, :, :, :, :] = result[c]
            # print('MP took {} sec.'.format(time.time()-ts))

        if self.multicam:
            X = X.reshape(
                (
                    self.batch_size,
                    len(self.camnames[first_exp]),
                    X.shape[1],
                    X.shape[2],
                    X.shape[3],
                    X.shape[4],
                )
            )
            X = X.permute((0, 2, 3, 4, 5, 1))

            if self.channel_combo == "avg":
                X = torch.mean(X, axis=-1)

            # Randomly reorder the cameras fed into the first layer
            elif self.channel_combo == "random":
                X = X[:, :, :, :, :, torch.randperm(X.shape[-1])]

                X = X.transpose(4, 5).reshape(
                    (
                        X.shape[0],
                        X.shape[1],
                        X.shape[2],
                        X.shape[3],
                        X.shape[4] * X.shape[5],
                    )
                )
            else:
                X = X.transpose(4, 5).reshape(
                    (
                        X.shape[0],
                        X.shape[1],
                        X.shape[2],
                        X.shape[3],
                        X.shape[4] * X.shape[5],
                    )
                )
        else:
            # Then leave the batch_size and num_cams combined
            y_3d = y_3d.repeat(num_cams, 1, 1, 1, 1)

        # 3dprob is required for *training* MAX networks
        if self.mode == "3dprob":
            y_3d = y_3d.permute([0, 2, 3, 4, 1])

        if self.rotation:
            if self.expval:
                # First make X_grid 3d
                X_grid = torch.reshape(
                    X_grid,
                    (self.batch_size, self.nvox, self.nvox, self.nvox, 3),
                )

                X, X_grid = self.random_rotate(X, X_grid)
                # Need to reshape back to raveled version
                X_grid = torch.reshape(X_grid, (self.batch_size, -1, 3))
            else:
                X, y_3d = self.random_rotate(X, y_3d)

        if self.mono and self.n_channels_in == 3:
            # Convert from RGB to mono using the skimage formula. Drop the duplicated frames.
            # Reshape so RGB can be processed easily.
            X = torch.reshape(
                X,
                (
                    X.shape[0],
                    X.shape[1],
                    X.shape[2],
                    X.shape[3],
                    len(self.camnames[first_exp]),
                    -1,
                ),
            )
            X = (
                X[:, :, :, :, :, 0] * 0.2125
                + X[:, :, :, :, :, 1] * 0.7154
                + X[:, :, :, :, :, 2] * 0.0721
            )

        # Convert pytorch tensors back to numpy array
        ts = time.time()
        if torch.is_tensor(X):
            X = X.float().cpu().numpy()
        if torch.is_tensor(y_3d):
            y_3d = y_3d.cpu().numpy()
        # print('Numpy took {} sec'.format(time.time() - ts))

        if self.expval:
            if torch.is_tensor(X_grid):
                X_grid = X_grid.cpu().numpy()
            if self.var_reg:
                return (
                    [processing.preprocess_3d(X), X_grid],
                    [y_3d, torch.zeros((self.batch_size, 1))],
                )

            if self.norm_im:
                # y_3d is in coordinates here.
                return [processing.preprocess_3d(X), X_grid], y_3d
            else:
                return [X, X_grid], y_3d
        else:
            if self.norm_im:
                return processing.preprocess_3d(X), y_3d
            else:
                return X, y_3d


class DataGenerator_3Dconv_torch(DataGenerator):
    """Update generator class to resample from kmeans clusters after each epoch.
    Also handles data across multiple experiments

    Attributes:
        camera_params (Dict): Camera parameters dictionary.
        channel_combo (Text): Method for shuffling camera input order
        com3d (Dict): Dictionary of com3d data.
        COM_aug (bool): If True, augment the COM.
        crop_im (bool): If True, crop images.
        depth (bool): If True, appends voxel depth to sampled image features [DEPRECATED]
        device (torch.device): GPU device identifier
        dim_out_3d (Tuple): Dimensions of the 3D volume, in voxels
        distort (bool): If true, apply camera undistortion.
        expval (bool): If True, process an expected value network (AVG)
        gpu_id (Text): Identity of GPU to use.
        immode (Text): Toggles using 'video' or 'tif' files as image input [DEPRECATED]
        interp (Text): Interpolation method.
        labels_3d (Dict): Contains ground-truth 3D label coordinates.
        mode (Text): Toggles output label format to match MAX vs. AVG network requirements.
        multicam (bool): If True, formats data to work with multiple cameras as input.
        norm_im (bool): If True, normalize images.
        nvox (int): Number of voxels per box side
        rotation (bool): If True, use simple rotation augmentation.
        session (tf.compat.v1.InteractiveSession): tensorflow session.
        threadpool (Threadpool): threadpool object for parallelizing video loading
        tifdirs (List): Directories of .tifs
        var_reg (bool): If True, adds a variance regularization term to the loss function.
        vmax (int): Maximum box dim (relative to the COM)
        vmin (int): Minimum box dim (relative to the COM)
        vsize (float): Side length of one voxel
        predict_flag (bool): If True, use imageio for reading videos, rather than OpenCV
    """

    def __init__(
        self,
        list_IDs,
        labels,
        labels_3d,
        camera_params,
        clusterIDs,
        com3d,
        tifdirs,
        batch_size=32,
        dim_in=(32, 32, 32),
        n_channels_in=1,
        n_channels_out=1,
        out_scale=5,
        shuffle=True,
        camnames=[],
        crop_width=(0, 1024),
        crop_height=(20, 1300),
        vmin=-100,
        vmax=100,
        nvox=32,
        gpu_id="0",
        interp="linear",
        depth=False,
        channel_combo=None,
        mode="3dprob",
        samples_per_cluster=0,
        immode="tif",
        rotation=False,
        vidreaders=None,
        distort=True,
        expval=False,
        multicam=True,
        var_reg=False,
        COM_aug=None,
        crop_im=True,
        norm_im=True,
        chunks=3500,
        mono=False,
        mirror=False,
        predict_flag=False,
        exp_voxel_size = None
    ):
        """Initialize data generator.

        Args:
            list_IDs (List): List of sample Ids
            labels (Dict): Dictionary of labels
            labels_3d (Dict): Dictionary of 3d labels.
            camera_params (Dict): Camera parameters dictionary.
            clusterIDs (List): List of sample Ids
            com3d (Dict): Dictionary of com3d data.
            tifdirs (List): Directories of .tifs
            batch_size (int, optional): Batch size to generate
            dim_in (Tuple, optional): Input dimension
            n_channels_in (int, optional): Number of input channels
            n_channels_out (int, optional): Number of output channels
            out_scale (int, optional): Scale of the output gaussians.
            shuffle (bool, optional): If True, shuffle the samples.
            camnames (List, optional): List of camera names.
            crop_width (Tuple, optional): (first, last) pixels in image width
            crop_height (Tuple, optional): (first, last) pixels in image height
            vmin (int, optional): Minimum box dim (relative to the COM)
            vmax (int, optional): Maximum box dim (relative to the COM)
            nvox (int, optional): Number of voxels per box side
            gpu_id (Text, optional): Identity of GPU to use.
            interp (Text, optional): Interpolation method.
            depth (bool): If True, appends voxel depth to sampled image features [DEPRECATED]
            channel_combo (Text): Method for shuffling camera input order
            mode (Text): Toggles output label format to match MAX vs. AVG network requirements.
            samples_per_cluster (int, optional): Samples per cluster
            immode (Text): Toggles using 'video' or 'tif' files as image input [DEPRECATED]
            rotation (bool, optional): If True, use simple rotation augmentation.
            vidreaders (Dict, optional): Dict containing video readers.
            distort (bool, optional): If true, apply camera undistortion.
            expval (bool, optional): If True, process an expected value network (AVG)
            multicam (bool): If True, formats data to work with multiple cameras as input.
            var_reg (bool): If True, adds a variance regularization term to the loss function.
            COM_aug (bool, optional): If True, augment the COM.
            crop_im (bool, optional): If True, crop images.
            norm_im (bool, optional): If True, normalize images.
            chunks (int, optional): Size of chunks when using chunked mp4.
            mono (bool, optional): If True, use grayscale image.
            predict_flag (bool, optional): If True, use imageio for reading videos, rather than OpenCV
            exp_voxel_size (None or dict): If not None, use this voxel size for the expected value network.
        """
        DataGenerator.__init__(
            self,
            list_IDs,
            labels,
            clusterIDs,
            batch_size,
            dim_in,
            n_channels_in,
            n_channels_out,
            out_scale,
            shuffle,
            camnames,
            crop_width,
            crop_height,
            samples_per_cluster,
            vidreaders,
            chunks,
            mono,
            mirror,
            predict_flag,
        )
        self.vmin = vmin
        self.vmax = vmax
        self.nvox = nvox
        self.vsize = (vmax - vmin) / nvox
        self.dim_out_3d = (nvox, nvox, nvox)
        self.labels_3d = labels_3d
        self.camera_params = camera_params
        self.interp = interp
        self.depth = depth
        self.channel_combo = channel_combo
        print('Real voxel size: {}mm'.format(self.vmax - self.vmin))
        self.gpu_id = gpu_id
        self.mode = mode
        self.immode = immode
        self.tifdirs = tifdirs
        self.com3d = com3d
        self.rotation = rotation
        self.distort = distort
        self.expval = expval
        self.multicam = multicam
        self.var_reg = var_reg
        self.COM_aug = COM_aug
        self.crop_im = crop_im
        # If saving npy as uint8 rather than training directly, dont normalize
        self.norm_im = norm_im
        self.exp_voxel_size = exp_voxel_size

        # importing torch here allows other modes to run without pytorch installed
        gpu_id_tf = os.getenv('CUDA_VISIBLE_DEVICES').split(',')[int(gpu_id)]
        self.device = torch.device(f"cuda:{gpu_id_tf}")
        # self.device = torch.device('cpu')

        # self.threadpool = ThreadPool(len(self.camnames[0]))

        ts = time.time()
        # Limit GPU memory usage by Tensorflow to leave memory for PyTorch
        gpus = tf.config.list_physical_devices("GPU")
        # set visible device as the first GPU
        gpu_id_tf = os.getenv('CUDA_VISIBLE_DEVICES').split(',')[int(gpu_id)]
        # tf.config.experimental.set_visible_devices(gpus[int(gpu_id_tf)], "GPU")
        # # set memory growth to True
        # tf.config.experimental.set_memory_growth(gpus[int(gpu_id_tf)], True)

        # tf.config.set_visible_devices(gpus[int(gpu_id_tf)], 'GPU')
        for gpu_device in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu_device, True)
        # config = tf.compat.v1.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.45
        # config.gpu_options.allow_growth = True
        # self.session = tf.compat.v1.InteractiveSession(config=config, graph=tf.Graph())

        experimentIDs = set([int(ID.split("_")[0]) for ID in list_IDs])

        for experimentID in experimentIDs:
            for camname in self.camnames[experimentID]:
                # M only needs to be computed once for each camera
                K = self.camera_params[experimentID][camname]["K"]
                R = self.camera_params[experimentID][camname]["R"]
                t = self.camera_params[experimentID][camname]["t"]
                M = torch.as_tensor(
                    ops.camera_matrix(K, R, t), dtype=torch.float32
                )
                self.camera_params[experimentID][camname]["M"] = M

        print("Init took {} sec.".format(time.time() - ts))

    def __getitem__(self, index: int):
        """Generate one batch of data.

        Args:
            index (int): Frame index

        Returns:
            Tuple[np.ndarray, np.ndarray]: One batch of data X
                (np.ndarray): Input volume y 
                (np.ndarray): Target
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def rot90(self, X):
        """Rotate X by 90 degrees CCW.

        Args:
            X (np.ndarray): Volume

        Returns:
            X (np.ndarray): Rotated volume
        """
        X = X.permute(1, 0, 2, 3)
        X = X.flip(1)
        return X

    def rot180(self, X):
        """Rotate X by 180 degrees.

        Args:
            X (np.ndarray): Volume

        Returns:
            X (np.ndarray): Rotated volume
        """
        X = X.flip(0).flip(1)
        return X

    def project_grid(self, X_grid, camname, ID, experimentID, thisim):
        """Projects 3D voxel centers and sample images as projected 2D pixel coordinates

        Args:
            X_grid (np.ndarray): 3-D array containing center coordinates of each voxel.
            camname (Text): camera name
            ID (Text): string denoting a sample ID
            experimentID (int): identifier for a video recording session.

        Returns:
            np.ndarray: projected voxel centers, now in 2D pixels
        """
        ts = time.time()
        # Need this copy so that this_y does not change
        this_y = torch.as_tensor(
            torch.zeros(2,2) * torch.nan,
            dtype=torch.float32,
            device=self.device,
        ).round()

        if torch.all(torch.isnan(this_y)):
            com_precrop = torch.zeros_like(this_y[:, 0]) * torch.nan
        else:
            # For projecting points, we should not use this offset
            com_precrop = torch.mean(this_y, axis=1)
        if not self.crop_height[0] or not self.crop_width[0]:
            this_y[0, :] = this_y[0, :] - self.crop_width[0]
            this_y[1, :] = this_y[1, :] - self.crop_height[0]
        com = torch.mean(this_y, axis=1)

        return self.pj_grid_post(X_grid, camname, experimentID, thisim)

    def pj_grid_mirror(self, X_grid, camname, ID, experimentID, thisim):
        this_y = torch.as_tensor(
            [torch.nan, torch.nan, torch.nan],
            dtype=torch.float32,
            device=self.device,
        ).round()

        if torch.all(torch.isnan(this_y)):
            com_precrop = torch.zeros_like(this_y[:, 0]) * torch.nan
        else:
            # For projecting points, we should not use this offset
            com_precrop = torch.mean(this_y, axis=1)

        this_y[0, :] = this_y[0, :] - self.crop_width[0]
        this_y[1, :] = this_y[1, :] - self.crop_height[0]
        com = torch.mean(this_y, axis=1)

        if not self.mirror:
            raise Exception("Trying to project onto mirrored images without mirror being set properly")

        if self.camera_params[experimentID][camname]["m"] == 1:
            passim = thisim[-1::-1].copy()
        elif self.camera_params[experimentID][camname]["m"] == 0:
            passim = thisim.copy()
        else:
            raise Exception("Invalid mirror parameter, m, must be 0 or 1")

        return self.pj_grid_post(X_grid, camname, experimentID, passim)

    def pj_grid_post(self, X_grid, camname, experimentID, thisim):
        # separate the porjection and sampling into its own function so that
        # when mirror == True, this can be called directly
        ts = time.time()
        assert not self.crop_im
        proj_grid = ops.project_to2d_torch(
            X_grid, self.camera_params[experimentID][camname]["M"], self.device
        )
        # print('Project2d took {} sec.'.format(time.time() - ts))

        # ts = time.time()
        if self.distort:
            proj_grid = ops.distortPoints_torch(
                proj_grid[:, :2],
                self.camera_params[experimentID][camname]["K"],
                np.squeeze(self.camera_params[experimentID][camname]["RDistort"]),
                np.squeeze(self.camera_params[experimentID][camname]["TDistort"]),
                self.device,
            )
            proj_grid = proj_grid.transpose(0, 1)
            # print('Distort took {} sec.'.format(time.time() - ts))

        assert self.crop_width[0]==self.crop_height[0]==0
        proj_grid = proj_grid[:, :2]

        rgb = ops.sample_grid_torch(thisim, proj_grid, self.device, method=self.interp)
        # print('Sample grid {} sec.'.format(time.time() - ts))

        if (
             (self.channel_combo == "avg")
            or not self.crop_im
        ):
            X = rgb.permute(0, 2, 3, 4, 1)

        return X

    # TODO(nesting): There is pretty deep locigal nesting in this function,
    # might be useful to break apart
    def __data_generation(self, list_IDs_temp):
        """Generate data containing batch_size samples.
        X : (n_samples, *dim, n_channels)

        Args:
            list_IDs_temp (List): List of experiment Ids

        Returns:
            Tuple: Batch_size training samples
                X: Input volumes
                y_3d: Targets
                rotangle: Rotation angle
        Raises:
            Exception: Invalid generator mode specified.
        """
        # Initialization
        first_exp = int(self.list_IDs[0].split("_")[0])
        assert not self.depth
        X = torch.zeros(
            (
                self.batch_size * len(self.camnames[first_exp]),
                *self.dim_out_3d,
                self.n_channels_in,
            ),
            dtype=torch.uint8,
            device=self.device,
        )

        if self.mode == "3dprob":
            y_3d = torch.zeros(
                (self.batch_size, self.n_channels_out, *self.dim_out_3d),
                dtype=torch.float32,
                device=self.device,
            )
        elif self.mode == "coordinates":
            y_3d = torch.zeros(
                (self.batch_size, 3, self.n_channels_out),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            raise Exception("not a valid generator mode")

        sz = self.dim_out_3d[0] * self.dim_out_3d[1] * self.dim_out_3d[2]
        X_grid = torch.zeros(
            (self.batch_size, sz, 3),
            dtype=torch.float32,
            device=self.device,
        )


        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            ts = time.time()
            sampleID = int(ID.split("_")[1])
            experimentID = int(ID.split("_")[0])

            # Voxel size refine
            if self.exp_voxel_size:
                voxel_size = self.exp_voxel_size[experimentID]
                vmin, vmax = -voxel_size / 2, voxel_size / 2
                vsize = voxel_size / self.nvox
            else:
                vmin, vmax = self.vmin, self.vmax
                vsize = self.vsize

            # For 3D ground truth
            this_y_3d = torch.as_tensor(
                self.labels_3d[ID],
                dtype=torch.float32,
                device=self.device,
            )
            this_COM_3d = torch.as_tensor(
                self.com3d[ID], dtype=torch.float32, device=self.device
            )
            this_COM_3d[2] += torch.rand([], device=self.device) * 40 - 20  # random z-axis offset
            # Create and project the grid here,
            grids = torch.arange(vmin + vsize/2, vmax, vsize, 
                dtype=torch.float32,
                device=self.device)
            xgrid = grids + this_COM_3d[0]
            ygrid = grids + this_COM_3d[1]
            zgrid = grids + this_COM_3d[2]
            (x_coord_3d, y_coord_3d, z_coord_3d) = torch.meshgrid(
                xgrid, ygrid, zgrid
            )
            X_grid[i] = torch.stack(
                (
                    x_coord_3d.transpose(0, 1).flatten(),
                    y_coord_3d.transpose(0, 1).flatten(),
                    z_coord_3d.transpose(0, 1).flatten(),
                ),
                axis=1,
            )

            if self.mode == "3dprob":
                X_grid_i = X_grid[i].reshape(*x_coord_3d.shape, 3)
                for j in range(self.n_channels_out):
                    y_3d[i, j] = torch.exp(
                        - torch.sum((X_grid_i - this_y_3d[:,j])**2, dim=-1)
                        / (2 * self.out_scale ** 2)
                    )

            if self.mode == "coordinates":
                if this_y_3d.shape == y_3d[i].shape:
                    y_3d[i] = this_y_3d
                else:
                    msg = "Note: ignoring dimension mismatch in 3D labels"
                    warnings.warn(msg)

            # Compute projected images in parallel using multithreading
            ts = time.time()
            num_cams = len(self.camnames[experimentID])
            arglist = []
            if self.mirror:
                # Here we only load the video once, and then parallelize the projection
                # and sampling after mirror flipping. For setups that collect views
                # in a single imgae with the use of mirrors
                loadim = self.load_frame.load_vid_frame(
                    self.labels[ID]["frames"][self.camnames[experimentID][0]],
                    self.camnames[experimentID][0],
                    extension=self.extension,
                )[
                    self.crop_height[0]: self.crop_height[1],
                    self.crop_width[0]: self.crop_width[1],
                ]
                for c in range(num_cams):
                    arglist.append(
                        [X_grid[i], self.camnames[experimentID][c], ID, experimentID, loadim]
                    )
                # result = self.threadpool.starmap(self.pj_grid_mirror, arglist)
                result = [self.pj_grid_mirror(*arg) for arg in arglist]
            else:
                views = get_view_xywh_wrapper(len(self.camera_params[experimentID]))
                imfile = self.labels[ID]
                # im_canvas = imageio.imread(imfile)  #old
                im_canvas = nj.read(imfile)  #new
                # print('Load image tooks {} sec.'.format(time.time()-ts))
                ims = [im_canvas[y:y+h, x:x+w] for (x,y,w,h) in views]
                for c in range(num_cams):
                    arglist.append(
                        [X_grid[i], self.camnames[experimentID][c], ID, experimentID, ims[c]]
                    )
                # result = self.threadpool.starmap(self.project_grid, arglist)
                ts = time.time()
                result = [self.project_grid(*arg) for arg in arglist]
                # print('Project took {} sec.'.format(time.time()-ts))

            for c in range(num_cams):
                ic = c + i * len(self.camnames[experimentID])
                X[ic, :, :, :, :] = result[c]
            # print('MP took {} sec.'.format(time.time()-ts))

        if self.multicam:
            X = X.reshape(
                (
                    self.batch_size,
                    len(self.camnames[first_exp]),
                    X.shape[1],
                    X.shape[2],
                    X.shape[3],
                    X.shape[4],
                )
            )
            X = X.permute((0, 2, 3, 4, 5, 1))

            if self.channel_combo == "avg":
                X = torch.mean(X, axis=-1)

            # Randomly reorder the cameras fed into the first layer
            elif self.channel_combo == "random":
                X = X[:, :, :, :, :, torch.randperm(X.shape[-1])]

                X = X.transpose(4, 5).reshape(
                    (
                        X.shape[0],
                        X.shape[1],
                        X.shape[2],
                        X.shape[3],
                        X.shape[4] * X.shape[5],
                    )
                )
            else:
                X = X.transpose(4, 5).reshape(
                    (
                        X.shape[0],
                        X.shape[1],
                        X.shape[2],
                        X.shape[3],
                        X.shape[4] * X.shape[5],
                    )
                )
        else:
            # Then leave the batch_size and num_cams combined
            y_3d = y_3d.repeat(num_cams, 1, 1, 1, 1)

        # 3dprob is required for *training* MAX networks
        if self.mode == "3dprob":
            y_3d = y_3d.permute([0, 2, 3, 4, 1])

        if self.rotation:
            if self.expval:
                # First make X_grid 3d
                X_grid = torch.reshape(
                    X_grid,
                    (self.batch_size, self.nvox, self.nvox, self.nvox, 3),
                )

                X, X_grid = self.random_rotate(X, X_grid)
                # Need to reshape back to raveled version
                X_grid = torch.reshape(X_grid, (self.batch_size, -1, 3))
            else:
                X, y_3d = self.random_rotate(X, y_3d)

        if self.mono and self.n_channels_in == 3:
            # Convert from RGB to mono using the skimage formula. Drop the duplicated frames.
            # Reshape so RGB can be processed easily.
            X = torch.reshape(
                X,
                (
                    X.shape[0],
                    X.shape[1],
                    X.shape[2],
                    X.shape[3],
                    len(self.camnames[first_exp]),
                    -1,
                ),
            )
            X = (
                X[:, :, :, :, :, 0] * 0.2125
                + X[:, :, :, :, :, 1] * 0.7154
                + X[:, :, :, :, :, 2] * 0.0721
            )

        # Convert pytorch tensors back to numpy array
        ts = time.time()
        if torch.is_tensor(X):
            X = X.float().cpu().numpy()
        if torch.is_tensor(y_3d):
            y_3d = y_3d.cpu().numpy()
        # print('Numpy took {} sec'.format(time.time() - ts))

        if self.expval:
            if torch.is_tensor(X_grid):
                X_grid = X_grid.cpu().numpy()
            if self.var_reg:
                return (
                    [processing.preprocess_3d(X), X_grid],
                    [y_3d, torch.zeros((self.batch_size, 1))],
                )

            if self.norm_im:
                # y_3d is in coordinates here.
                return [processing.preprocess_3d(X), X_grid], y_3d
            else:
                return [X, X_grid], y_3d
        else:
            if self.norm_im:
                return processing.preprocess_3d(X), y_3d
            else:
                return X, y_3d

class DataGenerator_3Dconv_torch_video(DataGenerator_3Dconv_torch):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.vid = None
        assert self.shuffle == False, "Shuffle is not supported for video data"
        assert self.vidreaders is None, "Video readers are not supported for video data"
        assert not self.rotation, "Rotation is not supported for video data"
        self.ba_pose = matlab_pose_to_cv2_pose(self.camera_params[0])

    def set_video(self, videopath):
        assert os.path.exists(videopath), "Video file not found"
        self.vid = ffmpegcv.VideoCaptureNV(videopath, pix_fmt="rgb24")
        self.nframes = len(self.vid)

    def __len__(self):
        return int(np.ceil(self.nframes / self.batch_size))

    def __getitem__(self, index):
        indexes = np.arange(self.batch_size) + index * self.batch_size
        indexes = indexes[indexes < self.nframes]
        list_frames_temp = [f'0_{i}' for i in indexes]
        X, y = self.__data_generation(list_frames_temp)
        return X, y
    
    
    def __data_generation(self, list_IDs_temp):
        first_exp = int(self.list_IDs[0].split("_")[0])
        num_cams = len(self.camnames[first_exp])
        X = [[] for _ in range(self.batch_size * num_cams)]
        X_grid = [[] for _ in range(self.batch_size)]

        assert self.mode  == "coordinates"
        y_3d = torch.zeros(
            (self.batch_size, 3, self.n_channels_out),
            dtype=torch.float32,
            device=self.device,
        )

        for i, ID in enumerate(list_IDs_temp):
            sampleID = int(ID.split("_")[1])
            experimentID = int(ID.split("_")[0])

            # read frame image
            views = get_view_xywh_wrapper(len(self.camera_params[experimentID]))
            ret, im_canvas = self.vid.read()
            assert ret
            ims = [im_canvas[y:y+h, x:x+w] for (x,y,w,h) in views]
            coms_2d = ims_to_com2ds(ims)
            coms_3d = com2ds_to_com3d(coms_2d, self.ba_pose)

            # For 3D ground truth
            this_y_3d = torch.zeros(
                (3, 14),
                dtype=torch.float32,
                device=self.device,
            )
            this_COM_3d = torch.as_tensor(
                coms_3d, dtype=torch.float32, device=self.device
            )
            xgrid = torch.arange(
                self.vmin + this_COM_3d[0] + self.vsize / 2,
                this_COM_3d[0] + self.vmax,
                self.vsize,
                dtype=torch.float32,
                device=self.device,
            )
            ygrid = torch.arange(
                self.vmin + this_COM_3d[1] + self.vsize / 2,
                this_COM_3d[1] + self.vmax,
                self.vsize,
                dtype=torch.float32,
                device=self.device,
            )
            zgrid = torch.arange(
                self.vmin + this_COM_3d[2] + self.vsize / 2,
                this_COM_3d[2] + self.vmax,
                self.vsize,
                dtype=torch.float32,
                device=self.device,
            )
            (x_coord_3d, y_coord_3d, z_coord_3d) = torch.meshgrid(
                xgrid, ygrid, zgrid
            )
            if self.mode == "coordinates":
                if this_y_3d.shape == y_3d[i].shape:
                    y_3d[i] = this_y_3d
                else:
                    msg = "Note: ignoring dimension mismatch in 3D labels"
                    warnings.warn(msg)
            
            X_grid[i] = torch.stack(
                (
                    x_coord_3d.transpose(0, 1).flatten(),
                    y_coord_3d.transpose(0, 1).flatten(),
                    z_coord_3d.transpose(0, 1).flatten(),
                ),
                axis=1,
            )
            ts = time.time()
            
            arglist = []

            for c in range(num_cams):
                arglist.append(
                    [X_grid[i], self.camnames[experimentID][c], ID, experimentID, ims[c]]
                )
            result = [self.project_grid(*arg) for arg in arglist]
            # result = self.threadpool.starmap(self.project_grid, arglist)

            for c in range(num_cams):
                ic = c + i * len(self.camnames[experimentID])
                X[ic] = result[c].reshape(*self.dim_out_3d,
                                        self.n_channels_in)

        X = torch.stack(X).float()
        X_grid = torch.stack(X_grid).float()

        if self.multicam:
            X = X.reshape(
                (
                    self.batch_size,
                    len(self.camnames[first_exp]),
                    X.shape[1],
                    X.shape[2],
                    X.shape[3],
                    X.shape[4],
                )
            )
            X = X.permute((0, 2, 3, 4, 5, 1))

            if self.channel_combo == "avg":
                X = torch.mean(X, axis=-1)

            # Randomly reorder the cameras fed into the first layer
            elif self.channel_combo == "random":
                X = X[:, :, :, :, :, torch.randperm(X.shape[-1])]

                X = X.transpose(4, 5).reshape(
                    (
                        X.shape[0],
                        X.shape[1],
                        X.shape[2],
                        X.shape[3],
                        X.shape[4] * X.shape[5],
                    )
                )
            else:
                X = X.transpose(4, 5).reshape(
                    (
                        X.shape[0],
                        X.shape[1],
                        X.shape[2],
                        X.shape[3],
                        X.shape[4] * X.shape[5],
                    )
                )
        else:
            # Then leave the batch_size and num_cams combined
            y_3d = y_3d.repeat(num_cams, 1, 1, 1, 1)

        assert not self.rotation

        if self.mono and self.n_channels_in == 3:
            # Convert from RGB to mono using the skimage formula. Drop the duplicated frames.
            # Reshape so RGB can be processed easily.
            X = torch.reshape(
                X,
                (
                    X.shape[0],
                    X.shape[1],
                    X.shape[2],
                    X.shape[3],
                    len(self.camnames[first_exp]),
                    -1,
                ),
            )
            X = (
                X[:, :, :, :, :, 0] * 0.2125
                + X[:, :, :, :, :, 1] * 0.7154
                + X[:, :, :, :, :, 2] * 0.0721
            )
        ts = time.time()
        assert self.expval, "expval must be True for this dataset"
        if self.norm_im or (self.expval and self.var_reg):
            X = processing.preprocess_3d(X)
        if torch.is_tensor(X):
            X = X.float().cpu().numpy()
        if torch.is_tensor(y_3d):
            y_3d = y_3d.cpu().numpy()
        if torch.is_tensor(X_grid):
            X_grid = X_grid.cpu().numpy()
        # print('Numpy took {} sec'.format(time.time() - ts))

        return [X, X_grid], y_3d


class DataGenerator_3Dconv_torch_video_canvas(DataGenerator_3Dconv_torch):
    def set_video(self, videopath, gpu, pkldata):
        assert os.path.exists(videopath), "Video file not found"
        assert self.batch_size == 1, "Batch size must be 1 for video data"
        self.vid = CanvasReaderPannelMask(videopath, pkldata, gpu=gpu)
        self.nframes = len(self.vid)
        self.nclass = 2
        self.batch_size = self.nclass
        self.n_channels_in = 1 #gray

    def __len__(self):
        return self.nframes

    def __getitem__(self, index):
        indexes = np.arange(self.batch_size) + index * self.batch_size
        list_frames_temp = [f'0_{i}' for i in indexes]
        X, y = self.__data_generation(list_frames_temp)
        return X, y
    
    def __data_generation(self, list_IDs_temp):
        first_exp = int(self.list_IDs[0].split("_")[0])
        num_cams = len(self.camnames[first_exp])
        self.batch_size = self.nclass
        X = [None for _ in range(self.batch_size * num_cams)]
        X_grid = [None for _ in range(self.batch_size)]

        y_3d = torch.zeros(
            (self.batch_size, 3, self.n_channels_out),
            dtype=torch.float32,
            device=self.device,
        )

        # 1A. Mask out
        im_pannels_nclass = self.vid.read_canvas_mask_img_out()
        # 1B. No mask
        # ret, im_pannels_nclass_no_mask = self.vid.read()
        # im_pannels_nclass = [im_pannels_nclass_no_mask] * self.batch_size

        # Finished
        assert len(im_pannels_nclass) == self.batch_size == len(list_IDs_temp)
        assert not self.depth
        for i, ID in enumerate(list_IDs_temp):
            experimentID = int(ID.split("_")[0])

            # read frame image
            ims = im_pannels_nclass[i]
            if 'coms_3d' in self.vid.pkl_data:
                coms_3d = self.vid.pkl_data['coms_3d'][self.vid.iframe,i,:]
            else:
                coms_2d = ims_to_com2ds(ims)
                coms_3d = com2ds_to_com3d(coms_2d, self.ba_pose)

            this_COM_3d = torch.as_tensor(
                coms_3d, dtype=torch.float32, device=self.device
            )

            grids = torch.arange(self.vmin + self.vsize/2, self.vmax, self.vsize, 
                dtype=torch.float32,
                device=self.device)
            xgrid = grids + this_COM_3d[0]
            ygrid = grids + this_COM_3d[1]
            zgrid = grids + this_COM_3d[2]
            (x_coord_3d, y_coord_3d, z_coord_3d) = torch.meshgrid(
                xgrid, ygrid, zgrid
            )
            X_grid[i] = torch.stack(
                (
                    x_coord_3d.transpose(0, 1).flatten(),
                    y_coord_3d.transpose(0, 1).flatten(),
                    z_coord_3d.transpose(0, 1).flatten(),
                ),
                axis=1,
            )
            ts = time.time()
            
            arglist = []

            for c in range(num_cams):
                arglist.append(
                    [X_grid[i], self.camnames[experimentID][c], experimentID, ims[c]]
                )
            result = [self.pj_grid_post(*arg) for arg in arglist]

            for c in range(num_cams):
                ic = c + i * len(self.camnames[experimentID])
                X[ic] = result[c].reshape(*self.dim_out_3d,
                                        self.n_channels_in)

        X = torch.stack(X).float()
        X_grid = torch.stack(X_grid).float()

        assert self.multicam
        X = X.reshape(
            (
                self.batch_size,
                len(self.camnames[first_exp]),
                X.shape[1],
                X.shape[2],
                X.shape[3],
                X.shape[4],
            )
        )
        X = X.permute((0, 2, 3, 4, 5, 1))

        assert not self.channel_combo == "avg"
        assert not self.channel_combo == "random"

        X = X.transpose(4, 5).reshape(
            (
                X.shape[0],
                X.shape[1],
                X.shape[2],
                X.shape[3],
                X.shape[4] * X.shape[5],
            )
        )

        assert not self.rotation

        if self.mono and self.n_channels_in == 3:
            # Convert from RGB to mono using the skimage formula. Drop the duplicated frames.
            # Reshape so RGB can be processed easily.
            X = torch.reshape(
                X,
                (
                    X.shape[0],
                    X.shape[1],
                    X.shape[2],
                    X.shape[3],
                    len(self.camnames[first_exp]),
                    -1,
                ),
            )
            X = (
                X[:, :, :, :, :, 0] * 0.2125
                + X[:, :, :, :, :, 1] * 0.7154
                + X[:, :, :, :, :, 2] * 0.0721
            )
        ts = time.time()
        if self.expval:
            if self.norm_im or (self.expval and self.var_reg):
                X = processing.preprocess_3d(X)
            if torch.is_tensor(X):
                X = X.float().cpu().numpy()
            if torch.is_tensor(y_3d):
                y_3d = y_3d.cpu().numpy()
            if torch.is_tensor(X_grid):
                X_grid = X_grid.cpu().numpy()
            # print('Numpy took {} sec'.format(time.time() - ts))
            return [X, X_grid], y_3d
        elif self.mode == "3dprob":
            if self.norm_im:
                X = processing.preprocess_3d(X)
            X = X.float().cpu().numpy()
            X_grid = X_grid.reshape(X_grid.shape[0], *x_coord_3d.shape, X_grid.shape[-1])
            X_grid = X_grid.cpu().numpy()
            y_3d = np.zeros((self.batch_size, *self.dim_out_3d, self.n_channels_out),dtype='float32')
            return [X, X_grid], y_3d


class DataGenerator_3Dconv_torch_video_canvas_faster(DataGenerator_3Dconv_torch):
    def set_video(self, videopath, gpu, pkldata):
        assert os.path.exists(videopath), "Video file not found"
        assert self.batch_size == 1, "Batch size must be 1 for video data"
        self.vid = CanvasReaderPannelMask(videopath, pkldata, gpu=gpu)
        self.nframes = len(self.vid)
        self.nclass = 2
        self.batch_size = self.nclass
        self.n_channels_in = 1 #gray
        self.gpu = gpu
        self.device = torch.device(f"cuda:{gpu}")

        coms_3d = self.vid.pkl_data['coms_3d']
        coms_3d = coms_3d.reshape(-1, coms_3d.shape[-1])
        self.init_grids(coms_3d)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return self.nframes

    def __getitem__(self, index):
        indexes = np.arange(self.batch_size) + index * self.batch_size
        list_frames_temp = [f'0_{i}' for i in indexes]
        X, y = self.__data_generation(list_frames_temp)
        return X, y

    def __data_generation(self, list_IDs_temp):
        self.batch_size = self.nclass
        im_pannels_nclass = self.vid.read_canvas_mask_img_out()
        assert len(im_pannels_nclass) == self.batch_size == len(list_IDs_temp)
        assert not self.depth
        assert self.mode == "3dprob"
        assert self.mono

        X, xgrid_roi, y_3d = [], [], []
        for i, ID in enumerate(list_IDs_temp):
            experimentID = int(ID.split("_")[0])
            assert experimentID == 0
            ims = im_pannels_nclass[i]
            if 'coms_3d' in self.vid.pkl_data:
                com_3d = self.vid.pkl_data['coms_3d'][self.vid.iframe,i,:]
            else:
                coms_2d = ims_to_com2ds(ims)
                com_3d = com2ds_to_com3d(coms_2d, self.ba_pose)
            [X_each, xgrid_roi_each], y_3d_each = self.quary_gridsample_by_com3d(com_3d, ims)
            X.append(X_each)
            xgrid_roi.append(xgrid_roi_each)
            y_3d.append(y_3d_each)
        # X = np.concatenate(X)
        # xgrid_roi = np.concatenate(xgrid_roi)
        # y_3d = np.concatenate(y_3d)
        X = np.stack(X, axis=0)
        return [X, xgrid_roi], y_3d

    def init_grids(self, com_3ds):
        vstep = (self.vmax - self.vmin) / self.nvox
        assert com_3ds.shape[1]==3
        Xaxis_min, Yaxis_min, Zaxis_min = com_3ds.min(axis=0)
        Xaxis_max, Yaxis_max, Zaxis_max = com_3ds.max(axis=0)
        xgrid = np.arange(Xaxis_min+1.2*self.vmin, Xaxis_max+1.2*self.vmax, vstep)
        ygrid = np.arange(Yaxis_min+1.2*self.vmin, Yaxis_max+1.2*self.vmax, vstep)
        zgrid = np.arange(Zaxis_min+1.2*self.vmin, Zaxis_max+1.2*self.vmax, vstep)
        (x_coord_3d, y_coord_3d, z_coord_3d) = np.meshgrid(xgrid, ygrid, zgrid)
        grid_flatten_3d = np.stack((x_coord_3d.ravel(), y_coord_3d.ravel(), z_coord_3d.ravel()),axis=1)
        experimentID=0

        camParams = [self.camera_params[experimentID][name] for name in self.camnames[experimentID]]
        proj_grid_voxel_ncam = []
        for camParam in camParams:
            proj_grid = ops.project_to2d(grid_flatten_3d, camParam["K"], camParam["R"], camParam["t"])
            proj_grid = proj_grid[:, :2]
            if self.distort:
                proj_grid = ops.distortPoints(
                    proj_grid,
                    camParam["K"],
                    np.squeeze(camParam["RDistort"]),
                    np.squeeze(camParam["TDistort"]),
                ).T
            proj_grid_voxel = np.reshape(proj_grid, [*x_coord_3d.shape, 2]).astype('float16')
            proj_grid_voxel_ncam.append(proj_grid_voxel)
            
        self.grid_1d = (xgrid, ygrid, zgrid)
        self.grid_coord_3d = np.stack([x_coord_3d, y_coord_3d, z_coord_3d], axis=-1).astype('float16')  #(nvox_y, nvox_x, nvox_z, 3)
        self.proj_grid_voxel_ncam = np.array(proj_grid_voxel_ncam).astype('int16')  #(ncam, nvox_y, nvox_x, nvox_z, 2)
        self.ncam = len(self.proj_grid_voxel_ncam)
        self.checked_imsize = False
        # self.threadpool = ThreadPool(self.ncam)
        # context = multiprocessing.get_context(method='fork')
        # self.processpool = context.Pool(self.ncam)

    def quary_gridsample_by_com3d(self, com_3d, ims):
        # input=gray, output=gray.
        assert len(com_3d)==3
        assert len(ims) == self.ncam
        com_index = np.array([np.searchsorted(self.grid_1d[i], com_3d[i], side='right')
                        for i in range(3)])  #(3,)
        com_range = np.floor(com_index[:,None] + [- self.nvox/2, self.nvox/2]).astype(int) #(3,2)

        xgrid_roi = self.grid_coord_3d[None,
                                        com_range[1][0]:com_range[1][1], 
                                        com_range[0][0]:com_range[0][1], 
                                        com_range[2][0]:com_range[2][1],
                                        :]   #(1, nvox_y, nvox_x, nvox_z, 3)

        if not self.checked_imsize:
            self.proj_grid_voxel_ncam_indravel = np.zeros([*self.proj_grid_voxel_ncam.shape[:-1], 1], dtype='int64')
            for i in range(self.ncam):
                im = ims[i]
                assert len(im.shape)==3 and im.shape[2]==1
                assert im.data.c_contiguous
                np.clip(self.proj_grid_voxel_ncam[i, ..., 0], 0, im.shape[1] - 1, out=self.proj_grid_voxel_ncam[i, ..., 0])
                np.clip(self.proj_grid_voxel_ncam[i, ..., 1], 0, im.shape[0] - 1, out=self.proj_grid_voxel_ncam[i, ..., 1])

                indravel = np.ravel_multi_index(np.moveaxis(self.proj_grid_voxel_ncam[i, ..., ::-1], -1, 0), im.shape[:2])
                self.proj_grid_voxel_ncam_indravel[i] = indravel[..., None]

            self.checked_imsize = True

        ## Case 1: sample by grid_sample
        # proj_grid_voxel_ncam_roi = self.proj_grid_voxel_ncam[:,
        #                                 com_range[1][0]:com_range[1][1], 
        #                                 com_range[0][0]:com_range[0][1], 
        #                                 com_range[2][0]:com_range[2][1],
        #                                 :]   #(ncam, nvox_y, nvox_x, nvox_z, 2)
        # arglist = [(ims[icam], proj_grid_voxel_ncam_roi[icam], 'nearest')
        #                 for icam in range(self.ncam)]
        # result = [sample_grid(*args) for args in arglist]

        ## Case 2: sample by grid_sample with faster index
        proj_grid_voxel_ncam_indravel_roi = self.proj_grid_voxel_ncam_indravel[:,
                                        com_range[1][0]:com_range[1][1], 
                                        com_range[0][0]:com_range[0][1], 
                                        com_range[2][0]:com_range[2][1],
                                        :]   #(ncam, nvox_y, nvox_x, nvox_z, 1)
        arglist = [(ims[icam], proj_grid_voxel_ncam_indravel_roi[icam], 'nearest')
                        for icam in range(self.ncam)]
        result = [sample_grid_ravel(*args) for args in arglist]

        X = np.empty(
            (1, self.ncam, self.nvox, self.nvox, self.nvox),
            dtype="float32",
        )
        for icam in range(self.ncam):
            r, g, b = result[icam]
            X[0, icam] = r

        assert self.norm_im
        X = processing.preprocess_3d(X)
        # X = X.cpu().numpy()
        
        y_3d = np.empty((1, self.nvox, self.nvox, self.nvox, 14), dtype="float32")
        X = np.moveaxis(X, 1, -1) #(1, self.nvox, self.nvox, self.nvox, self.ncam)
        return [X[0], xgrid_roi[0]], y_3d[0]


class DataGenerator_3Dconv_torch_video_canvas_faster_single(DataGenerator_3Dconv_torch_video_canvas_faster):
    def set_video(self, videopath, gpu, pkldata):
        assert os.path.exists(videopath), "Video file not found"
        assert self.batch_size == 1, "Batch size must be 1 for video data"
        self.vid = CanvasReaderPannel(videopath, pkldata, gpu=gpu)
        self.nframes = len(self.vid)
        self.nclass = 1
        self.batch_size = self.nclass
        self.n_channels_in = 1 #gray
        self.gpu = gpu
        self.device = torch.device(f"cuda:{gpu}")

        coms_3d = self.vid.pkl_data['coms_3d']
        coms_3d = coms_3d.reshape(-1, coms_3d.shape[-1])
        self.init_grids(coms_3d)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        indexes = np.arange(self.batch_size) + index * self.batch_size
        list_frames_temp = [f'0_{i}' for i in indexes]
        X, y = self.__data_generation(list_frames_temp)
        return X, y

    def __data_generation(self, list_IDs_temp):
        self.batch_size = self.nclass
        ret, ims_pannel = self.vid.read()
        assert ret
        im_pannels_nclass = np.array(ims_pannel)
        assert len(im_pannels_nclass) == self.batch_size == len(list_IDs_temp) == 1
        assert not self.depth
        assert self.mode == "3dprob"
        assert self.mono

        X, xgrid_roi, y_3d = [], [], []
        for i, ID in enumerate(list_IDs_temp):
            experimentID = int(ID.split("_")[0])
            assert experimentID == 0
            ims = im_pannels_nclass[i]
            if 'coms_3d' in self.vid.pkl_data:
                com_3d = self.vid.pkl_data['coms_3d'][self.vid.iframe,i,:]
            else:
                coms_2d = ims_to_com2ds(ims)
                com_3d = com2ds_to_com3d(coms_2d, self.ba_pose)
            [X_each, xgrid_roi_each], y_3d_each = self.quary_gridsample_by_com3d(com_3d, ims)
            X.append(X_each)
            xgrid_roi.append(xgrid_roi_each)
            y_3d.append(y_3d_each)
        X = np.stack(X, axis=0)
        return [X, xgrid_roi], y_3d


def sample_grid(im: Union[np.ndarray, torch.Tensor], 
                projPts: Union[np.ndarray, torch.Tensor], 
                method = "nearest"):
    """Transfer 3d features to 2d by projecting down to 2d grid.

    Use 2d interpolation to transfer features to 3d points that have
    projected down onto a 2d grid
    Note that function expects proj_grid to be flattened, so results should be
    reshaped after being returned
    """
    assert method == "nearest", "not a valid interpolation method"

    # But some of them could be rounded outside of the image
    proj_im = im[projPts[..., 1], projPts[..., 0], :]

    if im.shape[-1]==1:
        proj_r = proj_g = proj_b = proj_im[..., 0]
    else:
        proj_r = proj_im[..., 0]
        proj_g = proj_im[..., 1]
        proj_b = proj_im[..., 2]

    return proj_r, proj_g, proj_b


def sample_grid_ravel(im: np.ndarray, projPts_ravel: np.ndarray, method = "nearest"):
    assert method == "nearest", "not a valid interpolation method"
    assert im.data.c_contiguous
    proj_im = im.ravel()[projPts_ravel]

    if im.shape[-1]==1:
        proj_r = proj_g = proj_b = proj_im[..., 0]
    else:
        proj_r = proj_im[..., 0]
        proj_g = proj_im[..., 1]
        proj_b = proj_im[..., 2]

    return proj_r, proj_g, proj_b


def sample_grid_backup(im: np.ndarray, projPts: np.ndarray, method = "nearest"):
    """Transfer 3d features to 2d by projecting down to 2d grid.

    Use 2d interpolation to transfer features to 3d points that have
    projected down onto a 2d grid
    Note that function expects proj_grid to be flattened, so results should be
    reshaped after being returned
    """
    assert method == "nearest", "not a valid interpolation method"

    # But some of them could be rounded outside of the image
    projPts = (projPts[:, 1], projPts[:, 0])
    proj_r = im[:, :, 0]
    proj_r = proj_r[projPts]

    if im.shape[-1]==1:
        proj_g = proj_b = proj_r
    else:
        proj_g = im[:, :, 1]
        proj_g = proj_g[projPts]
        proj_b = im[:, :, 2]
        proj_b = proj_b[projPts]

    return proj_r, proj_g, proj_b


def random_continuous_rotation(X, y_3d, max_delta=5):
    """Rotates X and y_3d a random amount around z-axis.

    Args:
        X (np.ndarray): input image volume
        y_3d (np.ndarray): 3d target (for MAX network) or voxel center grid (for AVG network)
        max_delta (int, optional): maximum range for rotation angle.

    Returns:
        np.ndarray: rotated image volumes
        np.ndarray: rotated grid coordimates
    """
    rotangle = np.random.rand() * (2 * max_delta) - max_delta
    shearangle = (np.random.rand() * (2 * max_delta) - max_delta)/2
    X = tf.reshape(X, [X.shape[0], X.shape[1], X.shape[2], -1]).numpy()
    y_3d = tf.reshape(
        y_3d, [y_3d.shape[0], y_3d.shape[1], y_3d.shape[2], -1]
    ).numpy()
    for i in range(X.shape[0]):
        X[i] = tf.keras.preprocessing.image.apply_affine_transform(
            X[i],
            theta=rotangle,
            shear=shearangle,
            row_axis=0,
            col_axis=1,
            channel_axis=2,
            fill_mode="nearest",
            cval=0.0,
            order=1,
        )
        y_3d[i] = tf.keras.preprocessing.image.apply_affine_transform(
            y_3d[i],
            theta=rotangle,
            shear=shearangle,
            row_axis=0,
            col_axis=1,
            channel_axis=2,
            fill_mode="nearest",
            cval=0.0,
            order=1,
        )

    X = tf.reshape(X, [X.shape[0], X.shape[1], X.shape[2], X.shape[2], -1]).numpy()
    y_3d = tf.reshape(
        y_3d,
        [y_3d.shape[0], y_3d.shape[1], y_3d.shape[2], y_3d.shape[2], -1],
    ).numpy()

    return X, y_3d

# TODO(inherit): Several methods are repeated, consider inheriting from parent
class DataGenerator_3Dconv_frommem(keras.utils.Sequence):
    """Generate 3d conv data from memory.

    Attributes:
        augment_brightness (bool): If True, applies brightness augmentation
        augment_continuous_rotation (bool): If True, applies rotation augmentation in increments smaller than 90 degrees
        augment_hue (bool): If True, applies hue augmentation
        batch_size (int): Batch size
        bright_val (float): Brightness augmentation range (-bright_val, bright_val), as fraction of raw image brightness
        chan_num (int): Number of input channels
        data (np.ndarray): Image volumes
        expval (bool): If True, crafts input for an AVG network
        hue_val (float): Hue augmentation range (-hue_val, hue_val), as fraction of raw image hue range
        indexes (np.ndarray): Sample indices used for batch generation
        labels (Dict): Label dictionary
        list_IDs (List): List of sampleIDs
        nvox (int): Number of voxels in each grid dimension
        random (bool): If True, shuffles camera order for each batch
        rotation (bool): If True, applies rotation augmentation in 90 degree increments
        rotation_val (float): Range of angles used for continuous rotation augmentation
        shuffle (bool): If True, shuffle the samples before each epoch
        var_reg (bool): If True, returns input used for variance regularization
        xgrid (np.ndarray): For the AVG network, this contains the 3D grid coordinates
        n_rand_views (int): Number of reviews to sample randomly from the full set
        replace (bool): If True, samples n_rand_views with replacement
    """

    def __init__(
        self,
        list_IDs,
        data,
        labels,
        batch_size,
        rotation=True,
        random=True,
        chan_num=3,
        shuffle=True,
        expval=False,
        xgrid=None,
        var_reg=False,
        nvox=64,
        augment_brightness=True,
        augment_hue=True,
        augment_continuous_rotation=True,
        bright_val=0.05,
        hue_val=0.05,
        rotation_val=5,
        replace=True,
        n_rand_views=None,
        heatmap_reg=False,
        heatmap_reg_coeff=0.01,
    ):
        """Initialize data generator.

        Args:
            list_IDs (List): List of sampleIDs
            data data (np.ndarray): Image volumes
            labels (Dict): Label dictionar
            batch_size (int): batch size
            rotation (bool, optional): If True, applies rotation augmentation in 90 degree increments
            random (bool, optional): If True, shuffles camera order for each batch
            chan_num (int, optional): Number of input channels
            shuffle (bool, optional): If True, shuffle the samples before each epoch
            expval (bool, optional): If True, crafts input for an AVG network
            xgrid (None, optional): For the AVG network, this contains the 3D grid coordinates
            var_reg (bool, optional): If True, returns input used for variance regularization
            nvox (int, optional): Number of voxels in each grid dimension
            augment_brightness (bool, optional): If True, applies brightness augmentation
            augment_hue (bool, optional): If True, applies hue augmentation
            augment_continuous_rotation (bool, optional): If True, applies rotation augmentation in increments smaller than 90 degree
            bright_val (float, optional): brightness augmentation range (-bright_val, bright_val), as fraction of raw image brightness
            hue_val (float, optional): Hue augmentation range (-hue_val, hue_val), as fraction of raw image hue range
            rotation_val (float, optional): Range of angles used for continuous rotation augmentation
            n_rand_views (int, optional): Number of reviews to sample randomly from the full set
            replace (bool, optional): If True, samples n_rand_views with replacement
        """
        self.list_IDs = list_IDs
        self.data = data
        self.labels = labels
        self.rotation = rotation
        self.batch_size = batch_size
        self.random = random
        self.chan_num = chan_num
        self.shuffle = shuffle
        self.expval = expval
        self.augment_hue = augment_hue
        self.augment_continuous_rotation = augment_continuous_rotation
        self.augment_brightness = augment_brightness
        self.var_reg = var_reg
        self.xgrid = xgrid
        self.nvox = nvox
        self.bright_val = bright_val
        self.hue_val = hue_val
        self.rotation_val = rotation_val
        self.n_rand_views = n_rand_views
        self.replace = replace
        self.heatmap_reg = heatmap_reg
        self.heatmap_reg_coeff = heatmap_reg_coeff
        self.on_epoch_end()

    def __len__(self):
        """Denote the number of batches per epoch.

        Returns:
            int: Batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data.

        Args:
            index (int): Frame index

        Returns:
            Tuple[np.ndarray, np.ndarray]: One batch of data
                X (np.ndarray): Input volume
                y (np.ndarray): Target
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Update indexes after each epoch."""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def rot90(self, X):
        """Rotate X by 90 degrees CCW.

        Args:
            X (np.ndarray): Image volume or grid

        Returns:
            X (np.ndarray): Rotated image volume or grid
        """
        X = np.transpose(X, [1, 0, 2, 3])
        X = X[:, ::-1, :, :]
        return X

    def rot180(self, X):
        """Rotate X by 180 degrees.

        Args:
            X (np.ndarray): Image volume or grid

        Returns:
            X (np.ndarray): Rotated image volume or grid
        """
        X = X[::-1, ::-1, :, :]
        return X

    def random_rotate(self, X, y_3d):
        """Rotate each sample by 0, 90, 180, or 270 degrees.

        Args:
            X (np.ndarray): Image volumes
            y_3d (np.ndarray): 3D grid coordinates (AVG) or training target volumes (MAX)

        Returns:
            X (np.ndarray): Rotated image volumes
            y_3d (np.ndarray): Rotated 3D grid coordinates (AVG) or training target volumes (MAX)
        """
        rots = np.random.choice(np.arange(4), X.shape[0])
        for i in range(X.shape[0]):
            if rots[i] == 0:
                pass
            elif rots[i] == 1:
                # Rotate180
                X[i] = self.rot180(X[i])
                y_3d[i] = self.rot180(y_3d[i])
            elif rots[i] == 2:
                # Rotate90
                X[i] = self.rot90(X[i])
                y_3d[i] = self.rot90(y_3d[i])
            elif rots[i] == 3:
                # Rotate -90/270
                X[i] = self.rot90(X[i])
                X[i] = self.rot180(X[i])
                y_3d[i] = self.rot90(y_3d[i])
                y_3d[i] = self.rot180(y_3d[i])

        return X, y_3d

    def visualize(self, original, augmented):
        """Plots example image after augmentation

        Args:
            original (np.ndarray): image before augmentation
            augmented (np.ndarray): image after augmentation.
        """
        import matplotlib.pyplot as plt

        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.title("Original image")
        plt.imshow(original)

        plt.subplot(1, 2, 2)
        plt.title("Augmented image")
        plt.imshow(augmented)
        plt.show()
        input("Press Enter to continue...")

    def do_augmentation(self, X, X_grid, y_3d):
        """Applies augmentation

        Args:
            X (np.ndarray): image volumes
            X_grid (np.ndarray): 3D grid coordinates
            y_3d (np.ndarray): training targets

        Returns:
            X (np.ndarray): Augemented image volumes
            X_grid (np.ndarray): 3D grid coordinates
            y_3d (np.ndarray): Training targets
        """
        if self.rotation:
            if self.expval:
                # First make X_grid 3d
                X_grid = np.reshape(
                    X_grid,
                    (self.batch_size, self.nvox, self.nvox, self.nvox, 3),
                )
                X, X_grid = self.random_rotate(X.copy(), X_grid.copy())
                # Need to reshape back to raveled version
                X_grid = np.reshape(X_grid, (self.batch_size, -1, 3))
            else:
                X, y_3d = self.random_rotate(X.copy(), y_3d.copy())

        if np.random.random() < 0.5 and not self.expval:
            flipidx = [0,2,1,3,4,5,8,9,6,7,12,13,10,11]
            X= np.flip(X, axis=1)
            y_3d = np.flip(y_3d, axis=1)
            y_3d = y_3d[...,flipidx]

        if np.random.random() < 0.5 and self.chan_num==1:  # random deactivate a camera
            blackind = np.random.choice(X.shape[-1], 1)
            X[...,blackind] = np.min(X[0,0,...,0])

        if self.augment_continuous_rotation:
            if self.expval:
                # First make X_grid 3d
                X_grid = np.reshape(
                    X_grid,
                    (self.batch_size, self.nvox, self.nvox, self.nvox, 3),
                )
                X, X_grid = random_continuous_rotation(
                    X.copy(), X_grid.copy(), self.rotation_val
                )
                # Need to reshape back to raveled version
                X_grid = np.reshape(X_grid, (self.batch_size, -1, 3))
            else:
                X, y_3d = random_continuous_rotation(
                    X.copy(), y_3d.copy(), self.rotation_val
                )

        if self.augment_hue and self.chan_num == 3:
            for n_cam in range(int(X.shape[-1] / self.chan_num)):
                channel_ids = np.arange(
                    n_cam * self.chan_num,
                    n_cam * self.chan_num + self.chan_num,
                )
                X[..., channel_ids] = tf.image.random_hue(
                    X[..., channel_ids], self.hue_val
                )
        elif self.augment_hue:
            warnings.warn(
                "Trying to augment hue with an image that is not RGB. Skipping."
            )

        if self.augment_brightness:
            for n_cam in range(int(X.shape[-1] / self.chan_num)):
                channel_ids = np.arange(
                    n_cam * self.chan_num,
                    n_cam * self.chan_num + self.chan_num,
                )
                X[..., channel_ids] = tf.image.random_brightness(
                    X[..., channel_ids], self.bright_val
                )

        return X, X_grid, y_3d

    def do_random(self, X):
        """Randomly re-order camera views

        Args:
            X (np.ndarray): image volumes

        Returns:
            X (np.ndarray): Shuffled image volumes
        """
        if self.random:
            X = np.reshape(X,
                           (X.shape[0],
                            X.shape[1],
                            X.shape[2],
                            X.shape[3],
                            self.chan_num,
                            -1),
                           order='F')
            X = X[:, :, :, :, :, np.random.permutation(X.shape[-1])]
            X = np.reshape(X,
                           (X.shape[0],
                            X.shape[1],
                            X.shape[2],
                            X.shape[3],
                            X.shape[4]*X.shape[5]), order='F')

        if self.n_rand_views is not None:
            # Select a set of cameras randomly with replacement.
            X = np.reshape(X,
                           (X.shape[0],
                            X.shape[1],
                            X.shape[2],
                            X.shape[3],
                            self.chan_num,
                            -1),
                           order='F')
            if self.replace:
                X = X[..., np.random.randint(X.shape[-1], size=(self.n_rand_views,))]
            else:
                if not self.random:
                    raise Exception("For replace=False for n_rand_views, random must be turned on")
                X = X[:, :, :, :, :, :self.n_rand_views]
            X = np.reshape(X,
                           (X.shape[0],
                            X.shape[1],
                            X.shape[2],
                            X.shape[3],
                            X.shape[4]*X.shape[5]),
                           order='F')

        return X

    def get_max_gt_ind(self, X_grid, y_3d):
        """Uses the gt label position to find the index of the voxel corresponding to it.
        Used for heatmap regularization.
        """

        diff = np.sum((X_grid[:, :, :, np.newaxis] - y_3d[:, np.newaxis, :, :])**2, axis=2)
        inds = np.argmin(diff, axis=1)
        grid_d = int(np.round(X_grid.shape[1]**(1/3)))
        inds = np.unravel_index(inds, (grid_d, grid_d, grid_d))
        return np.stack(inds, axis=1)


    def __data_generation(self, list_IDs_temp):
        """Generate data containing batch_size samples.
        X : (n_samples, *dim, n_channels)

        Args:
            list_IDs_temp (List): List of experiment Ids

        Returns:
            Tuple: Batch_size training samples
                X: Input volumes
                y_3d: Targets
        Raises:
            Exception: For replace=False for n_rand_views, random must be turned on.
        """
        # Initialization

        X = np.zeros((self.batch_size, *self.data.shape[1:]))
        y_3d = np.zeros((self.batch_size, *self.labels.shape[1:]))

        # Only used when 
        if self.expval:
            X_grid = np.zeros((self.batch_size, *self.xgrid.shape[1:]))
        else:
            X_grid = None
            
        for i, ID in enumerate(list_IDs_temp):
            X[i] = self.data[ID].copy()
            y_3d[i] = self.labels[ID]
            if self.expval:
                X_grid[i] = self.xgrid[ID]

        X, X_grid, y_3d = self.do_augmentation(X, X_grid, y_3d)

        # Randomly re-order, if desired
        X = self.do_random(X)

        if self.expval:
            if self.heatmap_reg:
                return [X, X_grid, self.get_max_gt_ind(X_grid, y_3d)], [y_3d,
                    self.heatmap_reg_coeff*np.ones((self.batch_size, y_3d.shape[-1]), dtype='float32')]
            return [X, X_grid], y_3d
        else:
            return X, y_3d

class DataGenerator_3Dconv_npy(DataGenerator_3Dconv_frommem):
    """Generates 3d conv data from npy files.

        Attributes:
        augment_brightness (bool): If True, applies brightness augmentation
        augment_continuous_rotation (bool): If True, applies rotation augmentation in increments smaller than 90 degrees
        augment_hue (bool): If True, applies hue augmentation
        batch_size (int): Batch size
        bright_val (float): Brightness augmentation range (-bright_val, bright_val), as fraction of raw image brightness
        chan_num (int): Number of input channels
        labels_3d (Dict): training targets
        expval (bool): If True, crafts input for an AVG network
        hue_val (float): Hue augmentation range (-hue_val, hue_val), as fraction of raw image hue range
        indexes (np.ndarray): Sample indices used for batch generation
        list_IDs (List): List of sampleIDs
        nvox (int): Number of voxels in each grid dimension
        random (bool): If True, shuffles camera order for each batch
        rotation (bool): If True, applies rotation augmentation in 90 degree increments
        rotation_val (float): Range of angles used for continuous rotation augmentation
        shuffle (bool): If True, shuffle the samples before each epoch
        var_reg (bool): If True, returns input used for variance regularization
        n_rand_views (int): Number of reviews to sample randomly from the full set
        replace (bool): If True, samples n_rand_views with replacement
        imdir (Text): Name of image volume npy subfolder
        griddir (Text): Name of grid volumw npy subfolder
        mono (bool): If True, return monochrome image volumes
        sigma (float): For MAX network, size of target Gaussian (mm)
        cam1 (bool): If True, prepares input for training a single camea network
        prefeat (bool): If True, prepares input for a network performing volume feature extraction before fusion
        npydir (Dict): path to each npy volume folder for each recording (i.e. experiment)
    """

    def __init__(self,
                 list_IDs,
                 labels_3d,
                 npydir,
                 batch_size,
                 rotation=True,
                 random=False,
                 chan_num=3,
                 shuffle=True,
                 expval=False,
                 var_reg=False,
                 imdir='image_volumes',
                 griddir='grid_volumes',
                 nvox=64,
                 n_rand_views=None,
                 mono=False,
                 cam1=False,
                 replace=True,
                 prefeat=False,
                 sigma=10,
                 augment_brightness=True,
                 augment_hue=True,
                 augment_continuous_rotation=True,
                 bright_val=0.05,
                 hue_val=0.05,
                 rotation_val=5,
                 heatmap_reg=False,
                 heatmap_reg_coeff=0.01,
                 ):
        """Generates 3d conv data from npy files.

        Args:
            list_IDs (List): List of sampleIDs
            labels_3d (Dict): training targets
            npydir (Dict): path to each npy volume folder for each recording (i.e. experiment)
            batch_size (int): Batch size
            rotation (bool, optional): If True, applies rotation augmentation in 90 degree increments
            random (bool, optional): If True, shuffles camera order for each batch
            chan_num (int, optional): Number of input channels
            shuffle (bool, optional): If True, shuffle the samples before each epoch
            expval (bool, optional): If True, crafts input for an AVG network
            var_reg (bool, optional): If True, returns input used for variance regularization
            imdir (Text, optional): Name of image volume npy subfolder
            griddir (Text, optional): Name of grid volumw npy subfolder
            nvox (int, optional): Number of voxels in each grid dimension
            n_rand_views (int, optional): Number of reviews to sample randomly from the full set
            mono (bool, optional): If True, return monochrome image volumes
            cam1 (bool, optional): If True, prepares input for training a single camea network
            replace (bool, optional): If True, samples n_rand_views with replacement
            prefeat (bool, optional): If True, prepares input for a network performing volume feature extraction before fusion
            sigma (float, optional): For MAX network, size of target Gaussian (mm)
            augment_brightness (bool, optional): If True, applies brightness augmentation
            augment_hue (bool, optional): If True, applies hue augmentation
            augment_continuous_rotation (bool, optional): If True, applies rotation augmentation in increments smaller than 90 degrees
            bright_val (float, optional): Brightness augmentation range (-bright_val, bright_val), as fraction of raw image brightness
            hue_val (float, optional): Hue augmentation range (-hue_val, hue_val), as fraction of raw image hue range
            rotation_val (float, optional): Range of angles used for continuous rotation augmentation
        """
        self.list_IDs = list_IDs
        self.labels_3d = labels_3d
        self.npydir = npydir
        self.rotation = rotation
        self.batch_size = batch_size
        self.random = random
        self.chan_num = chan_num
        self.shuffle = shuffle
        self.expval = expval
        self.var_reg = var_reg
        self.griddir = griddir
        self.imdir = imdir
        self.nvox = nvox
        self.n_rand_views = n_rand_views
        self.mono = mono
        self.cam1 = cam1
        self.replace = replace
        self.prefeat = prefeat
        self.sigma = sigma
        self.augment_hue = augment_hue
        self.augment_continuous_rotation = augment_continuous_rotation
        self.augment_brightness = augment_brightness
        self.bright_val = bright_val
        self.hue_val = hue_val
        self.rotation_val = rotation_val
        self.heatmap_reg = heatmap_reg
        self.heatmap_reg_coeff = heatmap_reg_coeff
        self.on_epoch_end()

    def __len__(self):
        """Denote the number of batches per epoch.

        Returns:
            int: Batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data.

        Args:
            index (int): Frame index

        Returns:
            Tuple[np.ndarray, np.ndarray]: One batch of data
                X (np.ndarray): Input volume
                y (np.ndarray): Target
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Update indexes after each epoch."""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            print("SHUFFLING DATA INDICES")
            np.random.shuffle(self.indexes)

    def rot90(self, X):
        # Rotate 90
        X = np.transpose(X, [1, 0, 2, 3])
        X = X[:, ::-1, :, :]

        return X

    def rot180(self,X):
        #Rotate 180
        X = X[::-1, ::-1, :, :]

        return X

    def random_rotate(self, X, y_3d):
        """
        Rotate each sample by 0, 90, 180, or 270 degrees
        """
        rots = np.random.choice(np.arange(4), X.shape[0])

        for i in range(X.shape[0]):
            if rots[i]==0:
                pass
            elif rots[i]==1:
                #Rotate180
                X[i] = self.rot180(X[i])
                y_3d[i] = self.rot180(y_3d[i])
            elif rots[i]==2:
                #Rotate90
                X[i] = self.rot90(X[i])
                y_3d[i] = self.rot90(y_3d[i])
            elif rots[i]==3:
                #Rotate -90/270
                X[i] = self.rot90(X[i])
                X[i] = self.rot180(X[i])
                y_3d[i] = self.rot90(y_3d[i])
                y_3d[i] = self.rot180(y_3d[i])
            else:
                raise Exception("Failed to rotate properly")

        return X, y_3d

    def __data_generation(self, list_IDs_temp):
        """Generate data containing batch_size samples.
        X : (n_samples, *dim, n_channels)

        Args:
            list_IDs_temp (List): List of experiment Ids

        Returns:
            Tuple: Batch_size training samples
                X: Input volumes
                y_3d or y_3d_max: Targets
        Raises:
            Exception: For replace=False for n_rand_views, random must be turned on.
        """

        # Initialization

        X = []
        y_3d = []
        X_grid = []

        for i, ID in enumerate(list_IDs_temp):
            # Need to look up the experiment ID to get the correct directory
            IDkey = ID.split("_")
            eID = int(IDkey[0])
            sID = IDkey[1]

            X.append(np.load(os.path.join(self.npydir[eID],
                                          self.imdir,
                                          '0_' + sID + '.npy')).astype('float32'))

            y_3d.append(self.labels_3d[ID])
            X_grid.append(np.load(os.path.join(self.npydir[eID],
                                               self.griddir,
                                               '0_' + sID + '.npy')))

        X = np.stack(X)
        y_3d = np.stack(y_3d)

        X_grid = np.stack(X_grid)
        if not self.expval:
            y_3d_max = np.zeros((self.batch_size,
                                 self.nvox,
                                 self.nvox,
                                 self.nvox,
                                 y_3d.shape[-1]))

        if not self.expval:
            X_grid = np.reshape(X_grid, (-1,
                                         self.nvox,
                                         self.nvox,
                                         self.nvox,
                                         3))
            for gridi in range(X_grid.shape[0]):
                x_coord_3d = X_grid[gridi, :, :, :, 0]
                y_coord_3d = X_grid[gridi, :, :, :, 1]
                z_coord_3d = X_grid[gridi, :, :, :, 2]
                for j in range(y_3d_max.shape[-1]):
                        y_3d_max[gridi, :, :, :, j] = \
                            np.exp(-((y_coord_3d-y_3d[gridi, 1, j])**2 +
                                   (x_coord_3d-y_3d[gridi, 0, j])**2 +
                                   (z_coord_3d-y_3d[gridi, 2, j])**2)/(2*self.sigma**2))

        if self.mono and self.chan_num == 3:
            # Convert from RGB to mono using the skimage formula. Drop the duplicated frames.
            # Reshape so RGB can be processed easily.
            X = np.reshape(
                X,
                (
                    X.shape[0],
                    X.shape[1],
                    X.shape[2],
                    X.shape[3],
                    self.chan_num,
                    -1,
                ),
                order="F",
            )
            X = (
                X[:, :, :, :, 0] * 0.2125
                + X[:, :, :, :, 1] * 0.7154
                + X[:, :, :, :, 2] * 0.0721
            )

        ncam = int(X.shape[-1]//self.chan_num)

        X, X_grid, y_3d = self.do_augmentation(X, X_grid, y_3d)

        # Randomly re-order, if desired
        X = self.do_random(X)

        if self.cam1:
            # collapse the cameras to the batch dimensions.
            X = np.reshape(X,
                           (X.shape[0],
                            X.shape[1],
                            X.shape[2],
                            X.shape[3],
                            self.chan_num,
                            -1),
                           order='F')
            X = np.transpose(X, [0, 5, 1, 2, 3, 4])
            X = np.reshape(X,
                           (-1,
                            X.shape[2],
                            X.shape[3],
                            X.shape[4],
                            X.shape[5]))
            if self.expval:
                y_3d = np.tile(y_3d, [ncam, 1, 1])
                X_grid = np.tile(X_grid, [ncam, 1, 1])
            else:
                y_3d = np.tile(y_3d, [ncam, 1, 1, 1, 1])

        X = processing.preprocess_3d(X)

        XX = []
        if self.prefeat:
            for ix in range(ncam):
                XX.append(X[..., ix*self.chan_num:(ix+1)*self.chan_num])
            X = XX

        if self.expval:
            if not self.prefeat:
                X = [X]
            X = X + [X_grid]

        if self.expval:
            if self.heatmap_reg:
                return [X, X_grid, self.get_max_gt_ind(X_grid, y_3d)], [y_3d,
                    self.heatmap_reg_coeff*np.ones((self.batch_size, y_3d.shape[-1]), dtype='float32')]
            return X, y_3d
        else:
            return X, y_3d_max
