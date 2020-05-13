"""Kmeans generator for keras."""
import os
import numpy as np
from tensorflow import keras
from dannce.engine import processing as processing
from dannce.engine import ops as ops
import imageio
import warnings
import time

import tensorflow as tf
from multiprocessing.dummy import Pool as ThreadPool

# import multiprocessing as mp
# from joblib import Parallel, delayed

class DataGenerator(keras.utils.Sequence):
    """Generate data for Keras."""

    def __init__(
        self, list_IDs, labels, clusterIDs, batch_size=32,
        dim_in=(32, 32, 32), n_channels_in=1,
        n_channels_out=1, out_scale=5, shuffle=True, camnames=[],
        crop_width=(0, 1024), crop_height=(20, 1300),
        samples_per_cluster=0, training=True,
        vidreaders=None, chunks=3500):
        """Initialize Generator."""
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
        self.training = training
        self._N_VIDEO_FRAMES = chunks
        self.on_epoch_end()

        if self.vidreaders is not None:
            self.extension = '.' + list(
                vidreaders[camnames[0][0]].keys())[0].rsplit('.')[-1]

        assert len(self.list_IDs) == len(self.clusterIDs)

    def __len__(self):
        """Denote the number of batches per epoch."""
        if self.training:
            return self.samples_per_cluster * len(
                np.unique(self.clusterIDs)) // self.batch_size
        else:
            return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        """Update indexes after each epoch."""
        if self.training:
            print('Regnerating samples from clusters...')
            self.indexes = []

            uClusterID = np.unique(self.clusterIDs)
            for i in range(len(uClusterID)):
                clusterID = uClusterID[i]
                inds = np.where(self.clusterIDs == clusterID)[0]

                # draw randomly from the cluster
                inds_ = np.random.choice(inds, size=(self.samples_per_cluster,))

                self.indexes = self.indexes + list(inds_)
            self.indexes = np.array(self.indexes)
        else:
            self.indexes = np.arange(len(self.list_IDs))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def load_vid_frame(self, ind, camname, preload=True, extension='.mp4'):
        """Load video frame from a single camera.

        This is currently implemented for handling only one camera as input
        """

        pixelformat = "yuv420p"
        input_params = ["-hwaccel", "nvdec", "-c:v", "h264_cuvid", 
            "-hwaccel_device", self.gpuID]
        output_params = []

        fname = str(
           self. _N_VIDEO_FRAMES * int(np.floor(ind / self._N_VIDEO_FRAMES))) + extension
        frame_num = int(ind % self._N_VIDEO_FRAMES)
        keyname = os.path.join(camname, fname)
        if preload:
            return self.vidreaders[camname][keyname].get_data(frame_num).astype('uint8')
        else:
            vid = imageio.get_reader(self.vidreaders[camname][keyname],
                pixelformat=pixelformat,
                input_params=input_params, 
                output_params=output_params)
            im = vid.get_data(frame_num).astype('uint8') # keep as uint8 to save bandwidth
            # Without a sleep here, ffmpeg can hang on video close
            # time.sleep(0.01)
            vid.close()
            return im


class DataGenerator_3Dconv_kmeans(DataGenerator):
    """Update generator class to resample from kmeans clusters after each epoch.

    Also handles data across multiple experiments
    """

    def __init__(
        self, list_IDs, labels, labels_3d, camera_params, clusterIDs, com3d,
        tifdirs, batch_size=32, dim_in=(32, 32, 32), n_channels_in=1,
        n_channels_out=1, out_scale=5, shuffle=True,
        camnames=[], crop_width=(0, 1024), crop_height=(20, 1300),
        vmin=-100, vmax=100, nvox=32, gpuID = '0',
        interp='linear', depth=False, channel_combo=None, mode='3dprob',
        preload=True, samples_per_cluster=0, immode='tif', training=True,
        rotation=False, pregrid=None, pre_projgrid=None, stamp=False,
        vidreaders=None, distort=False, expval=False, multicam=True,
        var_reg=False, COM_aug=None, crop_im=True, norm_im=True, chunks=3500):
        """Initialize data generator."""
        DataGenerator.__init__(
            self, list_IDs, labels, clusterIDs, batch_size, dim_in,
            n_channels_in, n_channels_out, out_scale, shuffle,
            camnames, crop_width, crop_height,
            samples_per_cluster, training, vidreaders, chunks)
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
        self.preload = preload
        self.immode = immode
        self.tifdirs = tifdirs
        self.com3d = com3d
        self.rotation = rotation
        self.pregrid = pregrid
        self.pre_projgrid = pre_projgrid
        self.stamp = stamp
        self.distort = distort
        self.expval = expval
        self.multicam = multicam
        self.var_reg = var_reg
        self.COM_aug = COM_aug
        self.crop_im = crop_im
        # If saving npy as uint8 rather than training directly, dont normalize
        self.norm_im = norm_im
        self.gpuID = gpuID

        if self.pregrid is not None:
            # Then we need to save our world size for later use
            # we expect pregride to be (coord_x,coord_y,coord_z)
            self.worldsize = np.min(pregrid[0])

        if self.stamp:
            # To save time, "stamp" a 3d gaussian at each marker position
            (x_coord_3d, y_coord_3d, z_coord_3d) = \
                np.meshgrid(
                    np.arange(self.worldsize, -self.worldsize, self.vsize),
                    np.arange(self.worldsize, -self.worldsize, self.vsize),
                    np.arange(self.worldsize, -self.worldsize, self.vsize))

            self.stamp_ = np.exp(
                -((y_coord_3d - 0)**2 + (x_coord_3d - 0)**2 +
                  (z_coord_3d - 0)**2) / (2 * self.out_scale**2))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        indexes = \
            self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def rot90(self, X):
        """Rotate X by 90 degrees CCW."""
        X = np.transpose(X, [1, 0, 2, 3])
        X = X[:, ::-1, :, :]
        return X

    def rot180(self, X):
        """Rotate X by 180 degrees."""
        X = X[::-1, ::-1, :, :]
        return X

    def random_rotate(self, X, y_3d, log=False):
        """Rotate each sample by 0, 90, 180, or 270 degrees.

        log indicates whether to return the rotation pattern (for saving) as well
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

    def fetch_grid(self, c):
        """Return ROI from pregrid."""
        c0 = int((c[0] - (self.worldsize)) / self.vsize)
        c1 = int((c[1] - (self.worldsize)) / self.vsize)
        c2 = int((c[2] - (self.worldsize)) / self.vsize)

        x_coord_3d = self.pregrid[0][
            c0 - self.nvox // 2:c0 + self.nvox // 2,
            c0 - self.nvox // 2:c0 + self.nvox // 2,
            c0 - self.nvox // 2:c0 + self.nvox // 2]
        y_coord_3d = self.pregrid[1][
            c1 - self.nvox // 2:c1 + self.nvox // 2,
            c1 - self.nvox // 2:c1 + self.nvox // 2,
            c1 - self.nvox // 2:c1 + self.nvox // 2]
        z_coord_3d = self.pregrid[2][
            c2 - self.nvox // 2:c2 + self.nvox // 2,
            c2 - self.nvox // 2:c2 + self.nvox // 2,
            c2 - self.nvox // 2:c2 + self.nvox // 2]

        return x_coord_3d, y_coord_3d, z_coord_3d

    def fetch_projgrid(self, c, e, cam):
        """Return ROI from pre_projgrid."""
        c0 = int((c[0] - (self.worldsize)) / self.vsize)
        c1 = int((c[1] - (self.worldsize)) / self.vsize)
        c2 = int((c[2] - (self.worldsize)) / self.vsize)

        proj_grid = self.pre_projgrid[e][cam][
            c1 - self.nvox // 2:c1 + self.nvox // 2,
            c0 - self.nvox // 2:c0 + self.nvox // 2,
            c2 - self.nvox // 2:c2 + self.nvox // 2].copy()
        proj_grid = np.reshape(proj_grid, [-1, 3])

        return proj_grid

    # TODO(this vs self): The this_* naming convention is hard to read.
    # Consider revising
    # TODO(nesting): There is pretty deep locigal nesting in this function,
    # might be useful to break apart
    def __data_generation(self, list_IDs_temp):
        """Generate data containing batch_size samples.

        X : (n_samples, *dim, n_channels)
        """
        # Initialization
        first_exp = int(self.list_IDs[0].split('_')[0])

        X = np.zeros(
            (self.batch_size * len(self.camnames[first_exp]),
                *self.dim_out_3d, self.n_channels_in + self.depth),
            dtype='float32')

        if self.mode == '3dprob':
            print('3dprob')
            y_3d = np.zeros(
                (self.batch_size, self.n_channels_out, *self.dim_out_3d),
                dtype='float32')
        elif self.mode == 'coordinates':
            y_3d = np.zeros(
                (self.batch_size, 3, self.n_channels_out),
                dtype='float32')
        else:
            raise Exception("not a valid generator mode")

        if self.expval:
            sz = self.dim_out_3d[0] * self.dim_out_3d[1] * self.dim_out_3d[2]
            X_grid = np.zeros((self.batch_size, sz, 3), dtype='float32')

        # Generate data
        cnt = 0
        for i, ID in enumerate(list_IDs_temp):

            sampleID = int(ID.split('_')[1])
            experimentID = int(ID.split('_')[0])

            # For 3D ground truth
            this_y_3d = self.labels_3d[ID]
            this_COM_3d = self.com3d[ID]

            if self.COM_aug is not None:
                this_COM_3d = this_COM_3d.copy().ravel()
                this_COM_3d = this_COM_3d + self.COM_aug * 2 * np.random.rand(
                    len(this_COM_3d)) - self.COM_aug

            # Actually we need to create and project the grid here,
            # relative to the reference point (SpineM).
            if self.pregrid is None:
                xgrid = np.arange(
                    self.vmin + this_COM_3d[0] + self.vsize / 2,
                    this_COM_3d[0] + self.vmax, self.vsize)
                ygrid = np.arange(
                    self.vmin + this_COM_3d[1] + self.vsize / 2,
                    this_COM_3d[1] + self.vmax, self.vsize)
                zgrid = np.arange(
                    self.vmin + this_COM_3d[2] + self.vsize / 2,
                    this_COM_3d[2] + self.vmax, self.vsize)
                (x_coord_3d, y_coord_3d, z_coord_3d) = \
                    np.meshgrid(xgrid, ygrid, zgrid)
            else:
                (x_coord_3d, y_coord_3d, z_coord_3d) = \
                    self.fetch_grid(this_COM_3d)

            if self.mode == '3dprob':
                for j in range(self.n_channels_out):
                    if self.stamp:
                        # these are coordinates of each marker relative to COM
                        c = this_y_3d[:, j] - this_COM_3d
                        c0 = int((c[0] - (self.worldsize)) / self.vsize)
                        c1 = int((c[1] - (self.worldsize)) / self.vsize)
                        c2 = int((c[2] - (self.worldsize)) / self.vsize)
                        y_3d[i, j] = self.stamp_[
                            c1 - self.nvox // 2:c1 + self.nvox // 2,
                            c0 + self.nvox // 2:c0 - self.nvox // 2:-1,
                            c2 + self.nvox // 2:c2 - self.nvox // 2:-1]
                    else:
                        y_3d[i, j] = np.exp(
                            -((y_coord_3d - this_y_3d[1, j])**2 +
                              (x_coord_3d - this_y_3d[0, j])**2 +
                              (z_coord_3d - this_y_3d[2, j])**2) /
                            (2 * self.out_scale**2))
                        # When the voxel grid is coarse, we will likely miss
                        # the peak of the probability distribution, as it
                        # will lie somewhere in the middle of a large voxel.
                        # So here we renormalize to [~, 1]

            if self.mode == 'coordinates':
                if this_y_3d.shape == y_3d[i].shape:
                    y_3d[i] = this_y_3d
                else:
                    msg = "Note: ignoring dimension mismatch in 3D labels"
                    warnings.warn(msg)

            if self.expval:
                X_grid[i] = np.stack(
                    (x_coord_3d.ravel(), y_coord_3d.ravel(), z_coord_3d.ravel()),
                    axis=1)

            for camname in self.camnames[experimentID]:
                ts = time.time()
                # Need this copy so that this_y does not change
                this_y = np.round(self.labels[ID]['data'][camname]).copy()

                if np.all(np.isnan(this_y)):
                    com_precrop = np.zeros_like(this_y[:, 0]) * np.nan
                else:
                    # For projecting points, we should not use this offset
                    com_precrop = np.nanmean(this_y, axis=1)

                # Store sample
                # for pre-cropped tifs
                if self.immode == 'tif':
                    thisim = imageio.imread(
                        os.path.join(
                            self.tifdirs[experimentID],
                            camname,
                            '{}.tif'.format(sampleID)))

                # From raw video, need to crop
                elif self.immode == 'vid':
                    thisim = self.load_vid_frame(
                        self.labels[ID]['frames'][camname],
                        camname,
                        self.preload,
                        extension=self.extension)[
                            self.crop_height[0]:self.crop_height[1],
                            self.crop_width[0]:self.crop_width[1]]
                    # print("Decode frame took {} sec".format(time.time() - ts))
                    tss = time.time()

                # Load in the image file at the specified path
                elif self.immode == 'arb_ims':
                    thisim = imageio.imread(
                        self.tifdirs[experimentID] +
                        self.labels[ID]['frames'][camname][0] + '.jpg')

                if self.immode == 'vid' or self.immode == 'arb_ims':
                    this_y[0, :] = this_y[0, :] - self.crop_width[0]
                    this_y[1, :] = this_y[1, :] - self.crop_height[0]
                    com = np.nanmean(this_y, axis=1)

                    if self.crop_im:
                        if np.all(np.isnan(com)):
                            thisim = np.zeros(
                                (self.dim_in[1], self.dim_in[0], self.n_channels_in))
                        else:
                            thisim = processing.cropcom(
                                thisim, com, size=self.dim_in[0])

                # Project de novo or load in approximate (faster)
                # TODO(break up): This is hard to read, consider breaking up
                ts = time.time()
                if self.pre_projgrid is None:
                    proj_grid = ops.project_to2d(
                        np.stack(
                            (x_coord_3d.ravel(),
                             y_coord_3d.ravel(),
                             z_coord_3d.ravel()),
                            axis=1),
                        self.camera_params[experimentID][camname]['K'],
                        self.camera_params[experimentID][camname]['R'],
                        self.camera_params[experimentID][camname]['t'])
                else:
                    proj_grid = \
                        self.fetch_projgrid(this_COM_3d, experimentID, camname)

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
                        self.camera_params[experimentID][camname]['K'],
                        np.squeeze(self.camera_params[
                            experimentID][camname]['RDistort']),
                        np.squeeze(self.camera_params[
                            experimentID][camname]['TDistort'])).T
                # print("Distort took {} sec".format(time.time() - ts))

                # ts = time.time()
                if self.crop_im:
                    proj_grid = \
                        proj_grid[:, :2] - com_precrop + self.dim_in[0] // 2
                    # Now all coordinates should map properly to the image
                    # cropped around the COM
                else:
                    # Then the only thing we need to correct for is
                    # crops at the borders
                    proj_grid = proj_grid[:, :2]
                    proj_grid[:, 0] = proj_grid[:, 0] - self.crop_width[0]
                    proj_grid[:, 1] = proj_grid[:, 1] - self.crop_height[0]

                (r, g, b) = ops.sample_grid(thisim, proj_grid, method=self.interp)
                # print("Sample grid took {} sec".format(time.time() - ts))

                if ~np.any(np.isnan(com_precrop)) or (
                    self.channel_combo == 'avg') or not self.crop_im:

                    X[cnt, :, :, :, 0] = np.reshape(r, (self.nvox, self.nvox, self.nvox))
                    X[cnt, :, :, :, 1] = np.reshape(g, (self.nvox, self.nvox, self.nvox))
                    X[cnt, :, :, :, 2] = np.reshape(b, (self.nvox, self.nvox, self.nvox))
                    if self.depth:
                        X[cnt, :, :, :, 3] = np.reshape(d, (self.nvox, self.nvox, self.nvox))
                cnt = cnt + 1
                # print("Projection grid took {} sec".format(time.time() - tss))

        if self.multicam:
            X = np.reshape(X,
                (self.batch_size, len(self.camnames[first_exp]),
                    X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
            X = np.transpose(X, [0, 2, 3, 4, 5, 1])

            if self.channel_combo == 'avg':
                X = np.nanmean(X, axis=-1)
            # Randomly reorder the cameras fed into the first layer
            elif self.channel_combo == 'random':
                X = X[:, :, :, :, :, np.random.permutation(X.shape[-1])]
                X = np.reshape(X,
                    (X.shape[0], X.shape[1], X.shape[2], X.shape[3],
                        X.shape[4] * X.shape[5]),
                    order='F')
            else:
                X = np.reshape(X,
                    (X.shape[0], X.shape[1], X.shape[2], X.shape[3],
                        X.shape[4] * X.shape[5]),
                    order='F')
        else:
            # Then leave the batch_size and num_cams combined
            y_3d = np.tile(y_3d, [len(self.camnames[experimentID]), 1, 1, 1, 1])

        if self.mode == '3dprob':
            y_3d = np.transpose(y_3d, [0, 2, 3, 4, 1])

        if self.rotation:
            if self.expval:
                # First make X_grid 3d
                X_grid = np.reshape(X_grid,
                    (self.batch_size, self.nvox, self.nvox, self.nvox, 3))

                if self.norm_im:
                    X, X_grid = self.random_rotate(X, X_grid)
                else:
                    X, X_grid, rotate_log = self.random_rotate(X, X_grid, log=True)
                # Need to reshape back to raveled version
                X_grid = np.reshape(X_grid, (self.batch_size, -1, 3))
            else:
                if self.norm_im:
                    X, y_3d = self.random_rotate(X, y_3d)
                else:
                    X, y_3d, rotate_log = self.random_rotate(X, y_3d, log=True)

        # Then we also need to return the 3d grid center coordinates,
        # for calculating a spatial expected value
        # Xgrid is typically symmetric for 90 and 180 degree rotations
        # (when vmax and vmin are symmetric)
        # around the z-axis, so no need to rotate X_grid.
        if self.expval:
            if self.var_reg:
                return (
                    [processing.preprocess_3d(X), X_grid],
                    [y_3d, np.zeros((self.batch_size, 1))])

            if self.norm_im:
                # y_3d is in coordinates here.
                return [processing.preprocess_3d(X), X_grid], y_3d
            else:
                return [X, X_grid], [y_3d, rotate_log]
        else:
            if self.norm_im:
                return processing.preprocess_3d(X), y_3d
            else:
                return X, [y_3d, rotate_log]


class DataGenerator_3Dconv_kmeans_torch(DataGenerator):
    """Update generator class to resample from kmeans clusters after each epoch.
    Also handles data across multiple experiments
    """

    def __init__(
        self, list_IDs, labels, labels_3d, camera_params, clusterIDs, com3d,
        tifdirs, batch_size=32, dim_in=(32, 32, 32), n_channels_in=1,
        n_channels_out=1, out_scale=5, shuffle=True,
        camnames=[], crop_width=(0, 1024), crop_height=(20, 1300),
        vmin=-100, vmax=100, nvox=32, gpuID='0',
        interp='linear', depth=False, channel_combo=None, mode='3dprob',
        preload=True, samples_per_cluster=0, immode='tif', training=True,
        rotation=False, pregrid=None, pre_projgrid=None, stamp=False,
        vidreaders=None, distort=False, expval=False, multicam=True,
        var_reg=False, COM_aug=None, crop_im=True, norm_im=True, chunks=3500):
        """Initialize data generator."""
        DataGenerator.__init__(
            self, list_IDs, labels, clusterIDs, batch_size, dim_in,
            n_channels_in, n_channels_out, out_scale, shuffle,
            camnames, crop_width, crop_height,
            samples_per_cluster, training, vidreaders, chunks)
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
        self.gpuID = gpuID
        self.mode = mode
        self.preload = preload
        self.immode = immode
        self.tifdirs = tifdirs
        self.com3d = com3d
        self.rotation = rotation
        self.pregrid = pregrid
        self.pre_projgrid = pre_projgrid
        self.stamp = stamp
        self.distort = distort
        self.expval = expval
        self.multicam = multicam
        self.var_reg = var_reg
        self.COM_aug = COM_aug
        self.crop_im = crop_im
        # If saving npy as uint8 rather than training directly, dont normalize
        self.norm_im = norm_im

        # importing torch here allows other modes to run without pytorch installed
        self.torch = __import__('torch') 
        self.device = self.torch.device('cuda:' + self.gpuID)
        # self.device = self.torch.device('cpu')

        self.threadpool = ThreadPool(len(self.camnames[0]))

        ts = time.time()
        # Limit GPU memory usage by Tensorflow to leave memory for PyTorch
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.45
        config.gpu_options.allow_growth = True
        self.session = tf.compat.v1.InteractiveSession(config=config, graph = tf.Graph())

        if self.pregrid is not None:
            # Then we need to save our world size for later use
            # we expect pregride to be (coord_x,coord_y,coord_z)
            self.worldsize = np.min(pregrid[0])

        if self.stamp:
            # To save time, "stamp" a 3d gaussian at each marker position
            (x_coord_3d, y_coord_3d, z_coord_3d) = \
                self.torch.meshgrid(
                    self.torch.arange(self.worldsize, -self.worldsize, self.vsize),
                    self.torch.arange(self.worldsize, -self.worldsize, self.vsize),
                    self.torch.arange(self.worldsize, -self.worldsize, self.vsize))

            self.stamp_ = np.exp(
                -((y_coord_3d - 0)**2 + (x_coord_3d - 0)**2 +
                  (z_coord_3d - 0)**2) / (2 * self.out_scale**2))

        for i, ID in enumerate(list_IDs):
            experimentID = int(ID.split('_')[0])
            for camname in self.camnames[experimentID]:

                # M only needs to be computed once for each camera               
                K = self.camera_params[experimentID][camname]['K']
                R = self.camera_params[experimentID][camname]['R']
                t = self.camera_params[experimentID][camname]['t']
                M = self.torch.as_tensor(ops.camera_matrix(K,R,t),
                    dtype = self.torch.float32)
                self.camera_params[experimentID][camname]['M'] = M

        print("Init took {} sec.".format(time.time()-ts))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        indexes = \
            self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def rot90(self, X):
        """Rotate X by 90 degrees CCW."""
        X = X.permute(1, 0, 2, 3)        
        X = X.flip(1)
        return X

    def rot180(self, X):
        """Rotate X by 180 degrees."""
        X = X.flip(0).flip(1)
        return X

    def random_rotate(self, X, y_3d, log=False):
        """Rotate each sample by 0, 90, 180, or 270 degrees.

        log indicates whether to return the rotation pattern (for saving) as well
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

    def fetch_grid(self, c):
        """Return ROI from pregrid."""
        c0 = int((c[0] - (self.worldsize)) / self.vsize)
        c1 = int((c[1] - (self.worldsize)) / self.vsize)
        c2 = int((c[2] - (self.worldsize)) / self.vsize)

        x_coord_3d = self.pregrid[0][
            c0 - self.nvox // 2:c0 + self.nvox // 2,
            c0 - self.nvox // 2:c0 + self.nvox // 2,
            c0 - self.nvox // 2:c0 + self.nvox // 2]
        y_coord_3d = self.pregrid[1][
            c1 - self.nvox // 2:c1 + self.nvox // 2,
            c1 - self.nvox // 2:c1 + self.nvox // 2,
            c1 - self.nvox // 2:c1 + self.nvox // 2]
        z_coord_3d = self.pregrid[2][
            c2 - self.nvox // 2:c2 + self.nvox // 2,
            c2 - self.nvox // 2:c2 + self.nvox // 2,
            c2 - self.nvox // 2:c2 + self.nvox // 2]

        return x_coord_3d, y_coord_3d, z_coord_3d

    def fetch_projgrid(self, c, e, cam):
        """Return ROI from pre_projgrid."""
        c0 = int((c[0] - (self.worldsize)) / self.vsize)
        c1 = int((c[1] - (self.worldsize)) / self.vsize)
        c2 = int((c[2] - (self.worldsize)) / self.vsize)

        proj_grid = self.pre_projgrid[e][cam][
            c1 - self.nvox // 2:c1 + self.nvox // 2,
            c0 - self.nvox // 2:c0 + self.nvox // 2,
            c2 - self.nvox // 2:c2 + self.nvox // 2].copy()
        proj_grid = np.reshape(proj_grid, [-1, 3])
        proj_grid = self.torch.as_tensor(proj_grid, 
            dtype = self.torch.float32, 
            device = self.device)

        return proj_grid

    def project_grid(self, X_grid, camname, ID, experimentID):
        ts = time.time()
        # Need this copy so that this_y does not change
        this_y = self.torch.as_tensor(self.labels[ID]['data'][camname],
            dtype = self.torch.float32,
            device = self.device,
            ).round()
        
        if self.torch.all(self.torch.isnan(this_y)):
            com_precrop = self.torch.zeros_like(this_y[:, 0]) * self.torch.nan
        else:
            # For projecting points, we should not use this offset
            com_precrop = self.torch.mean(this_y, axis=1)

        this_y[0, :] = this_y[0, :] - self.crop_width[0]
        this_y[1, :] = this_y[1, :] - self.crop_height[0]
        com = self.torch.mean(this_y, axis=1)

        thisim = self.load_vid_frame(
            self.labels[ID]['frames'][camname],
            camname,
            self.preload,
            extension=self.extension)[
                self.crop_height[0]:self.crop_height[1],
                self.crop_width[0]:self.crop_width[1]]

        if self.crop_im:
            if self.torch.all(self.torch.isnan(com)):
                thisim = self.torch.zeros(
                    (self.dim_in[1], self.dim_in[0], self.n_channels_in),
                    dtype = self.torch.uint8,
                    device = self.device)
            else:
                thisim = processing.cropcom(
                    thisim, com, size=self.dim_in[0])
        # print('Frame loading took {} sec.'.format(time.time() - ts))

        ts = time.time()
        # Project de novo or load in approximate (faster)
        if self.pre_projgrid is None:
            proj_grid = ops.project_to2d_torch(
                X_grid,
                self.camera_params[experimentID][camname]['M'],
                self.device)
        # print('Project2d took {} sec.'.format(time.time() - ts))

        ts = time.time()
        if self.distort: # distort = True
            proj_grid = ops.distortPoints_torch(
                proj_grid[:, :2],
                self.camera_params[experimentID][camname]['K'],
                np.squeeze(self.camera_params[experimentID][camname]['RDistort']),
                np.squeeze(self.camera_params[experimentID][camname]['TDistort']),
                self.device)
            proj_grid = proj_grid.transpose(0,1)
            # print('Distort took {} sec.'.format(time.time() - ts))

        ts = time.time()
        if self.crop_im:
            proj_grid = \
                proj_grid[:, :2] - com_precrop + self.dim_in[0] // 2
            # Now all coordinates should map properly to the image cropped around the COM
        else:
            # Then the only thing we need to correct for is crops at the borders
            proj_grid = proj_grid[:, :2]
            proj_grid[:, 0] = proj_grid[:, 0] - self.crop_width[0]
            proj_grid[:, 1] = proj_grid[:, 1] - self.crop_height[0]

        rgb = ops.sample_grid_torch(thisim, proj_grid, self.device, method=self.interp)
        # print('Sample grid {} sec.'.format(time.time() - ts))

        if ~self.torch.any(self.torch.isnan(com_precrop)) or (self.channel_combo == 'avg') or not self.crop_im:
            X = rgb.permute(0,2,3,4,1)

        return X

    # TODO(this vs self): The this_* naming convention is hard to read.
    # Consider revising
    # TODO(nesting): There is pretty deep locigal nesting in this function,
    # might be useful to break apart
    def __data_generation(self, list_IDs_temp):
        """Generate data containing batch_size samples.
        X : (n_samples, *dim, n_channels)
        """
        # Initialization
        first_exp = int(self.list_IDs[0].split('_')[0])

        X = self.torch.zeros(
                (self.batch_size * len(self.camnames[first_exp]),
                *self.dim_out_3d, self.n_channels_in + self.depth), 
                dtype = self.torch.uint8,
                device = self.device)

        if self.mode == '3dprob':
            y_3d = self.torch.zeros(
                (self.batch_size, self.n_channels_out, *self.dim_out_3d),
                dtype = self.torch.float32,
                device = self.device)
        elif self.mode == 'coordinates':
            y_3d = self.torch.zeros(
                (self.batch_size, 3, self.n_channels_out),
                dtype = self.torch.float32,
                device = self.device)
        else:
            raise Exception("not a valid generator mode")

        sz = self.dim_out_3d[0] * self.dim_out_3d[1] * self.dim_out_3d[2]
        X_grid = self.torch.zeros((self.batch_size, sz, 3), 
            dtype = self.torch.float32,
            device = self.device)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            sampleID = int(ID.split('_')[1])
            experimentID = int(ID.split('_')[0])

            # For 3D ground truth
            this_y_3d = self.torch.as_tensor(self.labels_3d[ID], 
                dtype = self.torch.float32, 
                device = self.device)
            this_COM_3d = self.torch.as_tensor(self.com3d[ID], 
                dtype = self.torch.float32, 
                device = self.device)

            # Actually we need to create and project the grid here,
            # relative to the reference point (SpineM).
            if self.pregrid is None:
                xgrid = self.torch.arange(
                    self.vmin + this_COM_3d[0] + self.vsize / 2,
                    this_COM_3d[0] + self.vmax, self.vsize,
                    dtype = self.torch.float32,
                    device = self.device)
                ygrid = self.torch.arange(
                    self.vmin + this_COM_3d[1] + self.vsize / 2,
                    this_COM_3d[1] + self.vmax, self.vsize,
                    dtype = self.torch.float32,
                    device = self.device)
                zgrid = self.torch.arange(
                    self.vmin + this_COM_3d[2] + self.vsize / 2,
                    this_COM_3d[2] + self.vmax, self.vsize,
                    dtype = self.torch.float32,
                    device = self.device)
                (x_coord_3d, y_coord_3d, z_coord_3d) = \
                    self.torch.meshgrid(xgrid, ygrid, zgrid)
            else:
                (x_coord_3d, y_coord_3d, z_coord_3d) = \
                    self.fetch_grid(this_COM_3d)

            if self.mode == 'coordinates':
                if this_y_3d.shape == y_3d[i].shape:
                    y_3d[i] = this_y_3d
                else:
                    msg = "Note: ignoring dimension mismatch in 3D labels"
                    warnings.warn(msg)

            X_grid[i] = self.torch.stack(
                (x_coord_3d.transpose(0,1).flatten(), 
                y_coord_3d.transpose(0,1).flatten(), 
                z_coord_3d.transpose(0,1).flatten()),
                axis=1)

            # Compute projected images in parallel using multithreading 
            ts = time.time()
            num_cams = len(self.camnames[experimentID])
            arglist = []
            for c in range(num_cams):
                arglist.append([X_grid[i], self.camnames[experimentID][c], ID, experimentID])
            result = self.threadpool.starmap(self.project_grid, arglist)

            for c in range(num_cams):
                ic = c + i*len(self.camnames[experimentID])
                X[ic,:,:,:,:] = result[c]
            # print('MP took {} sec.'.format(time.time()-ts))

        if self.multicam:
            X = X.reshape(
                    (self.batch_size, len(self.camnames[first_exp]),
                    X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
            X = X.permute((0, 2, 3, 4, 5, 1))

            if self.channel_combo == 'avg':
                X = self.torch.mean(X, axis=-1)

            # Randomly reorder the cameras fed into the first layer
            elif self.channel_combo == 'random':
                X = X[:, :, :, :, :, self.torch.randperm(X.shape[-1])]

                X = X.transpose(4,5).reshape((X.shape[0], X.shape[1], 
                    X.shape[2], X.shape[3],
                    X.shape[4] * X.shape[5]))              
            else:
                X = X.transpose(4,5).reshape((X.shape[0], X.shape[1], 
                    X.shape[2], X.shape[3],
                    X.shape[4] * X.shape[5])) 

        # Then leave the batch_size and num_cams combined
        y_3d = y_3d.repeat(len(self.camnames[experimentID]),1,1,1,1)

        if self.rotation:
            if self.expval:
                # First make X_grid 3d
                X_grid = self.torch.reshape(X_grid,
                    (self.batch_size, self.nvox, self.nvox, self.nvox, 3))

                if self.norm_im:
                    X, X_grid = self.random_rotate(X, X_grid)
                else:
                    X, X_grid, rotate_log = self.random_rotate(X, X_grid, log=True)
                # Need to reshape back to raveled version
                X_grid = self.torch.reshape(X_grid, (self.batch_size, -1, 3))
            else:
                if self.norm_im:
                    X, y_3d = self.random_rotate(X, y_3d)
                else:
                    X, y_3d, rotate_log = self.random_rotate(X, y_3d, log=True)
        
        # Convert pytorch tensors back to numpy array
        ts = time.time()
        if self.torch.is_tensor(X):
            X = X.float().cpu().numpy()
        if self.torch.is_tensor(y_3d):
            y_3d = y_3d.cpu().numpy()
        # print('Numpy took {} sec'.format(time.time() - ts))

        if self.expval:
            if self.torch.is_tensor(X_grid):
                X_grid = X_grid.cpu().numpy()
            if self.var_reg:
                return (
                    [processing.preprocess_3d(X), X_grid],
                    [y_3d, self.torch.zeros((self.batch_size, 1))])

            if self.norm_im:
                # y_3d is in coordinates here.
                return [processing.preprocess_3d(X), X_grid], y_3d
            else:
                return [X, X_grid], [y_3d, rotate_log]
        else:
            if self.norm_im:
                return processing.preprocess_3d(X), y_3d
            else:
                return X, [y_3d, rotate_log]


class DataGenerator_3Dconv_kmeans_tf(DataGenerator):
    """Updated generator class to resample from kmeans clusters after each epoch.
    Uses tensorflow operations to accelerate generation of projection grid
    **Compatible with TF 2.0 and newer. Not compatible with 1.14 and previous versions.
    Also handles data across multiple experiments
    """

    def __init__(
        self, list_IDs, labels, labels_3d, camera_params, clusterIDs, com3d,
        tifdirs, batch_size=32, dim_in=(32, 32, 32), n_channels_in=1,
        n_channels_out=1, out_scale=5, shuffle=True,
        camnames=[], crop_width=(0, 1024), crop_height=(20, 1300),
        vmin=-100, vmax=100, nvox=32, gpuID='0',
        interp='linear', depth=False, channel_combo=None, mode='3dprob',
        preload=True, samples_per_cluster=0, immode='tif', training=True,
        rotation=False, pregrid=None, pre_projgrid=None, stamp=False,
        vidreaders=None, distort=False, expval=False, multicam=True,
        var_reg=False, COM_aug=None, crop_im=True, norm_im=True, chunks=3500):

        """Initialize data generator."""
        DataGenerator.__init__(
            self, list_IDs, labels, clusterIDs, batch_size, dim_in,
            n_channels_in, n_channels_out, out_scale, shuffle,
            camnames, crop_width, crop_height,
            samples_per_cluster, training, vidreaders, chunks)
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
        self.gpuID = gpuID
        self.mode = mode
        self.preload = preload
        self.immode = immode
        self.tifdirs = tifdirs
        self.com3d = com3d
        self.rotation = rotation
        self.pregrid = pregrid
        self.pre_projgrid = pre_projgrid
        self.stamp = stamp
        self.distort = distort
        self.expval = expval
        self.multicam = multicam
        self.var_reg = var_reg
        self.COM_aug = COM_aug
        self.crop_im = crop_im
        # If saving npy as uint8 rather than training directly, dont normalize
        self.norm_im = norm_im

        self.config = tf.compat.v1.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.8
        self.config.gpu_options.allow_growth = True
        self.session = tf.compat.v1.InteractiveSession(
            config=self.config)

        self.device = ('/GPU:' + self.gpuID)
        # self.device = ('/CPU:0')

        self.threadpool = ThreadPool(len(self.camnames[0]))

        with tf.device(self.device):
            ts = time.time()
            if self.pregrid is not None:
                # Then we need to save our world size for later use
                # we expect pregride to be (coord_x,coord_y,coord_z)
                self.worldsize = np.min(pregrid[0])

            if self.stamp:
                # To save time, "stamp" a 3d gaussian at each marker position
                (x_coord_3d, y_coord_3d, z_coord_3d) = \
                    np.meshgrid(
                        np.arange(self.worldsize, -self.worldsize, self.vsize),
                        np.arange(self.worldsize, -self.worldsize, self.vsize),
                        np.arange(self.worldsize, -self.worldsize, self.vsize))

                self.stamp_ = np.exp(
                    -((y_coord_3d - 0)**2 + (x_coord_3d - 0)**2 +
                      (z_coord_3d - 0)**2) / (2 * self.out_scale**2))

            for i, ID in enumerate(list_IDs):
                experimentID = int(ID.split('_')[0])
                for camname in self.camnames[experimentID]:

                    # M only needs to be computed once for each camera               
                    K = self.camera_params[experimentID][camname]['K']
                    R = self.camera_params[experimentID][camname]['R']
                    t = self.camera_params[experimentID][camname]['t']
                    self.camera_params[experimentID][camname]['M'] = \
                        np.array(ops.camera_matrix(K,R,t), dtype = 'float32')

        print("Init took {} sec.".format(time.time()-ts))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        indexes = \
            self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    @tf.function
    def rot90(self, X):
        """Rotate X by 90 degrees CCW."""
        X = tf.transpose(X, [1, 0, 2, 3])
        X = X[:, ::-1, :, :]
        return X

    @tf.function
    def rot180(self, X):
        """Rotate X by 180 degrees."""
        X = X[::-1, ::-1, :, :]
        return X

    def random_rotate(self, X, y_3d, log=False):
        """Rotate each sample by 0, 90, 180, or 270 degrees.
        log indicates whether to return the rotation pattern (for saving) as well
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

    def fetch_grid(self, c):
        """Return ROI from pregrid."""
        c0 = int((c[0] - (self.worldsize)) / self.vsize)
        c1 = int((c[1] - (self.worldsize)) / self.vsize)
        c2 = int((c[2] - (self.worldsize)) / self.vsize)

        x_coord_3d = self.pregrid[0][
            c0 - self.nvox // 2:c0 + self.nvox // 2,
            c0 - self.nvox // 2:c0 + self.nvox // 2,
            c0 - self.nvox // 2:c0 + self.nvox // 2]
        y_coord_3d = self.pregrid[1][
            c1 - self.nvox // 2:c1 + self.nvox // 2,
            c1 - self.nvox // 2:c1 + self.nvox // 2,
            c1 - self.nvox // 2:c1 + self.nvox // 2]
        z_coord_3d = self.pregrid[2][
            c2 - self.nvox // 2:c2 + self.nvox // 2,
            c2 - self.nvox // 2:c2 + self.nvox // 2,
            c2 - self.nvox // 2:c2 + self.nvox // 2]

        return x_coord_3d, y_coord_3d, z_coord_3d

    def fetch_projgrid(self, c, e, cam):
        """Return ROI from pre_projgrid."""
        c0 = int((c[0] - (self.worldsize)) / self.vsize)
        c1 = int((c[1] - (self.worldsize)) / self.vsize)
        c2 = int((c[2] - (self.worldsize)) / self.vsize)

        proj_grid = self.pre_projgrid[e][cam][
            c1 - self.nvox // 2:c1 + self.nvox // 2,
            c0 - self.nvox // 2:c0 + self.nvox // 2,
            c2 - self.nvox // 2:c2 + self.nvox // 2].copy()
        proj_grid = np.reshape(proj_grid, [-1, 3])

        return proj_grid

    def project_grid(self, X_grid, camname, ID, experimentID, device):
        ts = time.time()
        with tf.device(device):
            # Need this copy so that this_y does not change
            this_y = np.round(self.labels[ID]['data'][camname]).copy()

            if np.all(np.isnan(this_y)):
                com_precrop = np.zeros_like(this_y[:, 0]) * np.nan
            else:
                # For projecting points, we should not use this offset
                com_precrop = np.nanmean(this_y, axis=1)

            if self.immode == 'vid':
                ts = time.time()
                thisim = self.load_vid_frame(
                    self.labels[ID]['frames'][camname],
                    camname,
                    self.preload,
                    extension=self.extension)[
                        self.crop_height[0]:self.crop_height[1],
                        self.crop_width[0]:self.crop_width[1]]
                # print("Frame loading took {} sec.".format(time.time()-ts))

                this_y[0, :] = this_y[0, :] - self.crop_width[0]
                this_y[1, :] = this_y[1, :] - self.crop_height[0]
                com = np.nanmean(this_y, axis=1)

            if self.crop_im:
                # Cropping takes negligible time
                if np.all(np.isnan(com)):
                    thisim = np.zeros(
                        (self.dim_in[1], self.dim_in[0], self.n_channels_in),
                        dtype = 'uint8')
                else:
                    thisim = processing.cropcom(
                        thisim, com, size=self.dim_in[0])

            # Project de novo or load in approximate (faster)
            if self.pre_projgrid is None: # pre_projgrid = None
                ts = time.time()
                X_grid = tf.convert_to_tensor(X_grid)
                pts1 = tf.ones((X_grid.shape[0], 1), dtype='float32')
                projPts = tf.concat((X_grid, pts1), 1)
                M = tf.convert_to_tensor(self.camera_params[experimentID][camname]['M'], dtype = 'float32')
                proj_grid = ops.project_to2d_tf(projPts, M)
                # print("2D Project took {} sec.".format(time.time() - ts))

            if self.distort:
                ts = time.time()
                proj_grid = ops.distortPoints_tf(
                    proj_grid,
                    tf.constant(self.camera_params[experimentID][camname]['K'],
                        dtype = 'float32'),
                    tf.squeeze(tf.constant(self.camera_params[experimentID][camname]['RDistort'],
                        dtype = 'float32')),
                    tf.squeeze(tf.constant(self.camera_params[experimentID][camname]['TDistort'],
                        dtype = 'float32')))
                proj_grid = tf.transpose(proj_grid,(1,0))
                # print("tf Distort took {} sec.".format(time.time() - ts))            

            if self.crop_im:
                proj_grid = \
                    proj_grid - com_precrop + self.dim_in[0] // 2
                # Now all coordinates should map properly to the image
                # cropped around the COM
            else:
                # Then the only thing we need to correct for is crops at the borders
                proj_grid = proj_grid - tf.cast(tf.stack(self.crop_width[0],self.crop_height[0]), 'float32')

            ts = time.time()
            rgb = ops.sample_grid_tf(thisim, proj_grid, device, method=self.interp)
            # print("Sample grid tf took {} sec".format(time.time() - ts))

            X = tf.reshape(rgb, (self.nvox, self.nvox, self.nvox, 3))

            return X

    # TODO(this vs self): The this_* naming convention is hard to read.
    # Consider revising
    # TODO(nesting): There is pretty deep locigal nesting in this function,
    # might be useful to break apart
    def __data_generation(self, list_IDs_temp):
        """Generate data containing batch_size samples.

        X : (n_samples, *dim, n_channels)
        """
        # Initialization
        ts = time.time()
        first_exp = int(self.list_IDs[0].split('_')[0])

        with tf.device(self.device):
            if self.mode == '3dprob':
                y_3d = tf.zeros(
                    (self.batch_size, self.n_channels_out, *self.dim_out_3d),
                    dtype = 'float32')
            elif self.mode == 'coordinates':
                y_3d = tf.zeros(
                    (self.batch_size, 3, self.n_channels_out),
                    dtype = 'float32')
            else:
                raise Exception("not a valid generator mode")

            # sz = self.dim_out_3d[0] * self.dim_out_3d[1] * self.dim_out_3d[2]
            # X_grid = tf.zeros((self.batch_size, sz, 3), dtype = 'float32')

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            sampleID = int(ID.split('_')[1])
            experimentID = int(ID.split('_')[0])

            # For 3D ground truth
            this_y_3d = self.labels_3d[ID]
            this_COM_3d = self.com3d[ID]

            with tf.device(self.device):
                # Actually we need to create and project the grid here,
                # relative to the reference point (SpineM).
                if self.pregrid is None:
                    xgrid = tf.range(
                        self.vmin + this_COM_3d[0] + self.vsize / 2,
                        this_COM_3d[0] + self.vmax, self.vsize,
                        dtype = 'float32')
                    ygrid = tf.range(
                        self.vmin + this_COM_3d[1] + self.vsize / 2,
                        this_COM_3d[1] + self.vmax, self.vsize,
                        dtype = 'float32')
                    zgrid = tf.range(
                        self.vmin + this_COM_3d[2] + self.vsize / 2,
                        this_COM_3d[2] + self.vmax, self.vsize,
                        dtype = 'float32')
                    (x_coord_3d, y_coord_3d, z_coord_3d) = \
                        tf.meshgrid(xgrid, ygrid, zgrid)                    
                else:
                    (x_coord_3d, y_coord_3d, z_coord_3d) = \
                        self.fetch_grid(this_COM_3d)

                if self.mode == 'coordinates':
                    if this_y_3d.shape == y_3d.shape:
                        if i==0:
                            y_3d = tf.expand_dims(y_3d, 0)
                        else:
                            y_3d = tf.stack(y_3d, tf.expand_dims(this_y_3d,0), axis=0)
                    else:
                        msg = "Note: ignoring dimension mismatch in 3D labels"
                        warnings.warn(msg)

                xg = tf.stack(
                        (tf.keras.backend.flatten(x_coord_3d), 
                        tf.keras.backend.flatten(y_coord_3d), 
                        tf.keras.backend.flatten(z_coord_3d)), 
                        axis=1)

                if i == 0:
                    X_grid = tf.expand_dims(xg,0)
                else:
                    X_grid = tf.concat([X_grid, tf.expand_dims(xg,0)], axis=0)

                # print('Initialization took {} sec.'.format(time.time() - ts))
                if tf.executing_eagerly():
                    # Compute projection grids using multithreading
                    num_cams = int(len(self.camnames[experimentID]))
                    arglist = []
                    for c in range(num_cams):
                        arglist.append(
                            [xg, self.camnames[experimentID][c], ID, experimentID, self.device])
                    result = self.threadpool.starmap(self.project_grid, arglist)
                    for c in range(num_cams):
                        if i==0 and c==0:
                            X = tf.expand_dims(result[c],0)
                        else:
                            X = tf.concat([X, tf.expand_dims(result[c],0)], axis=0)
                else:
                    for c in range(num_cams):
                        if c==0:
                            X = tf.expand_dims(self.project_grid(
                                    xg, self.camnames[experimentID][c], ID, experimentID, self.device),0)
                        else:
                            X = tf.concat((X,
                                    tf.expand_dims(self.project_grid(
                                        xg, self.camnames[experimentID][c], ID, experimentID, self.device),0)),
                                    axis=0)

        ts = time.time()
        with tf.device(self.device):

            if self.multicam:
                X = tf.reshape(
                    X,
                    (self.batch_size, len(self.camnames[first_exp]),
                        X.shape[1], X.shape[2], X.shape[3], X.shape[4]))
                X = tf.transpose(X, [0, 2, 3, 4, 5, 1])

                if self.channel_combo == 'avg':
                    X = tf.mean(X, axis=-1)
                # Randomly reorder the cameras fed into the first layer
                elif self.channel_combo == 'random':
                    X = tf.transpose(X, [5, 0, 1, 2, 3, 4])
                    X = tf.random.shuffle(X)
                    X = tf.transpose(X, [1, 2, 3, 4, 0, 5])
                    X = tf.reshape(
                        X,
                        (X.shape[0], X.shape[1], X.shape[2], X.shape[3],
                            X.shape[4] * X.shape[5]))
                else:
                    X = tf.transpose(X, [0, 1, 2, 3, 5, 4])
                    X = tf.reshape(
                            X,
                            (X.shape[0], X.shape[1], X.shape[2], X.shape[3],
                                X.shape[4] * X.shape[5]))
            else:
                # Then leave the batch_size and num_cams combined
                y_3d = tf.tile(y_3d, [len(self.camnames[experimentID]), 1, 1, 1, 1])

            if self.interp == 'linear':
                # fix rotation issue for linear interpolation sample_grid method
                X = tf.squeeze(X)
                X = self.rot90(X[:, ::-1, :, :])
                X = self.rot180(X)
                X = tf.expand_dims(X,0)

            if self.mode == '3dprob':
                y_3d = tf.transpose(y_3d, [0, 2, 3, 4, 1])

            X = tf.cast(X, 'float32')
               
            if self.rotation:
                if self.expval:
                    # First make X_grid 3d
                    X_grid = tf.reshape(
                        X_grid,
                        (self.batch_size, self.nvox, self.nvox, self.nvox, 3))

                    if self.norm_im:
                        X, X_grid = self.random_rotate(X, X_grid)
                    else:
                        X, X_grid, rotate_log = self.random_rotate(X, X_grid, log=True)
                    # Need to reshape back to raveled version
                    X_grid = tf.reshape(X_grid, (self.batch_size, -1, 3))
                else:
                    if self.norm_im:
                        X, y_3d = self.random_rotate(X, y_3d)
                    else:
                        X, y_3d, rotate_log = self.random_rotate(X, y_3d, log=True)

        # Then we also need to return the 3d grid center coordinates,
        # for calculating a spatial expected value
        # Xgrid is typically symmetric for 90 and 180 degree rotations
        # (when vmax and vmin are symmetric)
        # around the z-axis, so no need to rotate X_grid.
            # ts = time.time()
            # Eager execution enabled in TF 2, tested in TF 2.0, 2.1, and 2.2
            if tf.executing_eagerly():
                X = X.numpy()
                y_3d = y_3d.numpy()
                X_grid = X_grid.numpy()
            else:
                # For compatibility with TF 1.14
                # Eager execution disabled on 1.14; enabling eager causes model to fail
                # Works on 1.14, but very slow. Graph grows in loop... 
                X = X.eval(session=self.session)
                y_3d = y_3d.eval(session=self.session)
                X_grid = X_grid.eval(session=self.session)

            # print('Eval took {} sec.'.format(time.time()-ts))

        # print('Wrap-up took {} sec.'.format(time.time()-ts))

        if self.expval:
            if self.var_reg:
                return (
                    [processing.preprocess_3d(X), X_grid],
                    [y_3d, np.zeros((self.batch_size, 1))])

            if self.norm_im:
                # y_3d is in coordinates here.
                return [processing.preprocess_3d(X), X_grid], y_3d
            else:
                return [X, X_grid], [y_3d, rotate_log]
        else:
            if self.norm_im:
                return processing.preprocess_3d(X), y_3d
            else:
                return X, [y_3d, rotate_log]


# TODO(inherit): Several methods are repeated, consider inheriting from parent
class DataGenerator_3Dconv_frommem(keras.utils.Sequence):
    """Generate 3d conv data from memory."""

    def __init__(
        self, list_IDs, data, labels, batch_size, rotation=True, random=True,
        chan_num=3, shuffle=True, expval=False, xgrid=None, var_reg=False, nvox=64):
        """Initialize data generator."""
        self.list_IDs = list_IDs
        self.data = data
        self.labels = labels
        self.rotation = rotation
        self.batch_size = batch_size
        self.random = random
        self.chan_num = 3
        self.shuffle = shuffle
        self.expval = expval
        self.var_reg = var_reg
        if self.expval:
            self.xgrid = xgrid
        self.nvox = nvox
        self.on_epoch_end()

    def __len__(self):
        """Denote the number of batches per epoch."""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        indexes = \
            self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

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
        """Rotate X by 90 degrees CCW."""
        X = np.transpose(X, [1, 0, 2, 3])
        X = X[:, ::-1, :, :]
        return X

    def rot180(self, X):
        """Rotate X by 180 degrees."""
        X = X[::-1, ::-1, :, :]
        return X

    def random_rotate(self, X, y_3d):
        """Rotate each sample by 0, 90, 180, or 270 degrees."""
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

    def __data_generation(self, list_IDs_temp):
        """Generate data containing batch_size samples."""
        # Initialization

        X = np.zeros((self.batch_size, *self.data.shape[1:]))
        y_3d = np.zeros((self.batch_size, *self.labels.shape[1:]))

        if self.expval:
            X_grid = np.zeros((self.batch_size, *self.xgrid.shape[1:]))

        for i, ID in enumerate(list_IDs_temp):
            X[i] = self.data[ID].copy()
            y_3d[i] = self.labels[ID]
            if self.expval:
                X_grid[i] = self.xgrid[ID]

        if self.rotation:
            if self.expval:
                # First make X_grid 3d
                X_grid = np.reshape(
                    X_grid,
                    (self.batch_size, self.nvox, self.nvox, self.nvox, 3))
                X, X_grid = self.random_rotate(X.copy(), X_grid.copy())
                # Need to reshape back to raveled version
                X_grid = np.reshape(X_grid, (self.batch_size, -1, 3))
            else:
                X, y_3d = self.random_rotate(X.copy(), y_3d.copy())

        # Randomly re-order, if desired
        if self.random:
            X = np.reshape(
                X,
                (X.shape[0], X.shape[1], X.shape[2],
                    X.shape[3], self.chan_num, -1),
                order='F')
            X = X[:, :, :, :, :, np.random.permutation(X.shape[-1])]
            X = np.reshape(
                X,
                (X.shape[0], X.shape[1], X.shape[2], X.shape[3],
                    X.shape[4] * X.shape[5]),
                order='F')

        if self.expval:
            if self.var_reg:
                return [X, X_grid], [y_3d, np.zeros(self.batch_size)]
            return [X, X_grid], y_3d
        else:
            return X, y_3d

class DataGenerator_3Dconv_multiviewconsistency(keras.utils.Sequence):
    """Generate 3d conv data from memory, with a multiview consistency objective"""

    def __init__(
        self, list_IDs, data, labels, batch_size, rotation=True, random=True,
        chan_num=3, shuffle=True, expval=False, xgrid=None, var_reg=False, nvox=64):
        """Initialize data generator."""
        self.list_IDs = list_IDs
        self.data = data
        self.labels = labels
        self.rotation = rotation
        self.batch_size = 1 # We expand the number of examples in each batch for the multi-view consistency loss
        self.random = random
        self.chan_num = 3
        self.shuffle = shuffle
        self.expval = expval
        self.var_reg = var_reg
        if self.expval:
            self.xgrid = xgrid
        self.nvox = nvox
        self.on_epoch_end()

    def __len__(self):
        """Denote the number of batches per epoch."""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        indexes = \
            self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

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

    def __data_generation(self, list_IDs_temp):
        """Generate data containing batch_size samples."""
        # Initialization

        X = np.zeros((self.batch_size*4, *self.data.shape[1:]))
        y_3d = np.zeros((self.batch_size*4, *self.labels.shape[1:]))

        for i, ID in enumerate(list_IDs_temp):
            X[-1] = self.data[ID].copy()
            y_3d = np.tile(self.labels[ID][np.newaxis, :, :, :, :], (4, 1, 1, 1, 1)) # We just take 4 copies of this

        # Now we need to sub-select camera pairs in each of the first 3 examples,
        # so
        # idx 0: Camera1/Camera2
        # idx 1: Camera2/Camera3
        # idx 2: Camera1/Camera3
        # idx 3: Camera1/Camera2/Camera3

        order_dict = {}
        order_dict[0] = [0, 0, 0, 1, 1, 1]
        order_dict[1] = [1, 1, 1, 2, 2, 2]
        order_dict[2] = [0, 0, 0, 2, 2, 2]

        for i in range(3):
            c1_2_X = X[-1].copy()

            c1_2_X = np.reshape(
                c1_2_X,
                (c1_2_X.shape[0], c1_2_X.shape[1],
                    c1_2_X.shape[2], self.chan_num, -1),
                order='F')
            c1_2_X = c1_2_X[:, :, :, :, order_dict[i]]
            X[i] = np.reshape(
                c1_2_X,
                (c1_2_X.shape[0], c1_2_X.shape[1], c1_2_X.shape[2],
                    c1_2_X.shape[3] * c1_2_X.shape[4]),
                order='F')

        return X, y_3d