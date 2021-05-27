"""Generator for 3d video images."""
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import os
from tensorflow.keras.applications.vgg19 import preprocess_input as pp_vgg19
import imageio
from dannce.engine import processing as processing
import scipy.io as sio
import warnings
import time
import matplotlib.pyplot as plt
from dannce.engine.video import LoadVideoFrame
from typing import Text, Tuple, List, Union, Dict

_DEFAULT_CAM_NAMES = [
    "CameraR",
    "CameraL",
    "CameraU",
    "CameraU2",
    "CameraS",
    "CameraE",
]
_EXEP_MSG = "Desired Label channels and ground truth channels do not agree"


class DataGenerator_downsample(keras.utils.Sequence):
    """Generate data for Keras."""

    def __init__(
        self,
        list_IDs,
        labels,
        vidreaders,
        batch_size=32,
        dim_in=(1024, 1280),
        n_channels_in=1,
        n_channels_out=1,
        out_scale=5,
        shuffle=True,
        camnames=_DEFAULT_CAM_NAMES,
        crop_width=(0, 1024),
        crop_height=(20, 1300),
        downsample=1,
        immode="video",
        labelmode="prob",
        dsmode="dsm",
        chunks=3500,
        multimode=False,
        mono=False,
        mirror=False,
        predict_flag=False,
    ):
        """Initialize generator.

        TODO(params_definitions)
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
        self.downsample = downsample
        self.dsmode = dsmode
        self.mirror = mirror
        self.on_epoch_end()

        if immode == "video":
            self.extension = (
                "." + list(vidreaders[camnames[0][0]].keys())[0].rsplit(".")[-1]
            )

        self.immode = immode
        self.labelmode = labelmode
        # self.chunks = int(chunks)
        self.multimode = multimode

        self._N_VIDEO_FRAMES = chunks

        self.mono = mono

        self.predict_flag = predict_flag
        self.load_frame = LoadVideoFrame(self._N_VIDEO_FRAMES,
                                         self.vidreaders,
                                         self.camnames,
                                         self.predict_flag)

    def __len__(self):
        """Denote the number of batches per epoch."""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
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

    def load_tif_frame(self, ind, camname):
        """Load frames in tif mode."""
        # In tif mode, vidreaders should just be paths to the tif directory
        return imageio.imread(
            os.path.join(self.vidreaders[camname], "{}.tif".format(ind))
        )

    def __data_generation(self, list_IDs_temp):
        """Generate data containing batch_size samples.

        # X : (n_samples, *dim, n_channels)
        """
        # Initialization
        if self.mirror:
            X = np.empty(
                (self.batch_size, *self.dim_in, self.n_channels_in),
                dtype="uint8",
            )            
        else:
            X = np.empty(
                (self.batch_size * len(self.camnames[0]), *self.dim_in, self.n_channels_in),
                dtype="uint8",
            )

        # We'll need to transpose this later such that channels are last,
        # but initializaing the array this ways gives us
        # more flexibility in terms of user-defined array sizes\
        if self.labelmode == "prob":
            y = np.empty(
                (
                    self.batch_size * len(self.camnames[0]),
                    self.n_channels_out,
                    *self.dim_out,
                ),
                dtype="float32",
            )
        else:
            # Just return the targets, without making a meshgrid later
            y = np.empty(
                (
                    self.batch_size * len(self.camnames[0]),
                    self.n_channels_out,
                    len(self.dim_out),
                ),
                dtype="float32",
            )

        # Generate data
        cnt = 0
        for i, ID in enumerate(list_IDs_temp):
            if "_" in ID:
                experimentID = int(ID.split("_")[0])
            else:
                # Then we only have one experiment
                experimentID = 0
            for _ci, camname in enumerate(self.camnames[experimentID]):
                # Store sample
                # TODO(Refactor): This section is tricky to read

                if not self.mirror or _ci == 0:
                    if self.immode == "video":
                        X[cnt] = self.load_frame.load_vid_frame(
                            self.labels[ID]["frames"][camname],
                            camname,
                            self.extension,
                        )[
                            self.crop_height[0] : self.crop_height[1],
                            self.crop_width[0] : self.crop_width[1],
                        ]
                    elif self.immode == "tif":
                        X[cnt] = self.load_tif_frame(
                            self.labels[ID]["frames"][camname], camname
                        )[
                            self.crop_height[0] : self.crop_height[1],
                            self.crop_width[0] : self.crop_width[1],
                        ]
                    else:
                        raise Exception("Not a valid image reading mode")

                # Labels will now be the pixel positions of each joint.
                # Here, we convert them to
                # probability maps with a numpy meshgrid operation
                this_y = np.round(self.labels[ID]["data"][camname])
                if self.immode == "video":
                    this_y[0, :] = this_y[0, :] - self.crop_width[0]
                    this_y[1, :] = this_y[1, :] - self.crop_height[0]
                else:
                    raise Exception(
                        "Unsupported image format. Needs to be video files."
                    )

                # For 2D, this_y should be size (2, 20)
                if this_y.shape[1] != self.n_channels_out:
                    # TODO(shape_exception):This should probably be its own
                    # class that inherits from base exception
                    raise Exception(_EXEP_MSG)

                if self.labelmode == "prob":
                    # Only do this if we actually need the labels --
                    # this is too slow otherwise
                    (x_coord, y_coord) = np.meshgrid(
                        np.arange(self.dim_out[1]), np.arange(self.dim_out[0])
                    )
                    for j in range(self.n_channels_out):
                        # I tested a version of this with numpy broadcasting,
                        # and looping was ~100ms seconds faster for making
                        # 20 maps
                        # In the future, a shortcut might be to "stamp" a
                        # truncated Gaussian pdf onto the images, centered
                        # at the peak
                        y[cnt, j] = np.exp(
                            -(
                                (y_coord - this_y[1, j]) ** 2
                                + (x_coord - this_y[0, j]) ** 2
                            )
                            / (2 * self.out_scale ** 2)
                        )
                else:
                    y[cnt] = this_y.T

                cnt = cnt + 1

        # Move channels last
        if self.labelmode == "prob":
            y = np.transpose(y, [0, 2, 3, 1])

            if self.mirror:
                # separate the batches from the cameras, and use the cameras as the numebr of channels 
                # to make a single-shot multi-target prediction from a single image
                y = np.reshape(y, (self.batch_size, len(self.camnames[0]), y.shape[1], y.shape[2]))
                y = np.transpose(y, [0, 2, 3, 1])
        else:
            # One less dimension when not training with probability map targets
            y = np.transpose(y, [0, 2, 1])

        if self.downsample > 1:
            X = processing.downsample_batch(X, fac=self.downsample, method=self.dsmode)
            if self.labelmode == "prob":
                y = processing.downsample_batch(
                    y, fac=self.downsample, method=self.dsmode
                )
                y /= np.max(np.max(y, axis=1), axis=1)[:, np.newaxis, np.newaxis, :]

        if self.mono and self.n_channels_in == 3:
            # Go from 3 to 1 channel using RGB conversion. This will also
            # work fine if there are just 3 channel grayscale
            X = X[:, :, :, 0] * 0.2125 + X[:, :, :, 1] * 0.7154 + X[:, :, :, 2] * 0.0721

            X = X[:, :, :, np.newaxis]

        if self.mono:
            # Just subtract the mean imagent BGR value, which is as close as we
            # get to vgg19 normalization
            X -= 114.67
        else:
            X = pp_vgg19(X)
        return X, y


class DataGenerator_downsample_multi_instance(keras.utils.Sequence):
    """Generate data for Keras."""

    def __init__(
        self,
        n_instances,
        list_IDs,
        labels,
        vidreaders,
        batch_size: int = 32,
        dim_in: Tuple = (1024, 1280),
        n_channels_in: int = 1,
        n_channels_out: int = 1,
        out_scale: int = 5,
        shuffle=True,
        camnames: List = _DEFAULT_CAM_NAMES,
        crop_width: Tuple = (0, 1024),
        crop_height: Tuple = (20, 1300),
        downsample: int = 1,
        immode: Text = "video",
        labelmode: Text = "prob",
        dsmode: Text = "dsm",
        chunks: int = 3500,
        multimode: bool = False,
        mono: bool = False,
        mirror: bool = False,
        predict_flag: bool = False,
    ):
        """Initialize generator.

        TODO(params_definitions)
        """
        self.n_instances = n_instances
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
        self.downsample = downsample
        self.dsmode = dsmode
        self.on_epoch_end()

        if immode == "video":
            self.extension = (
                "." + list(vidreaders[camnames[0][0]].keys())[0].rsplit(".")[-1]
            )

        self.immode = immode
        self.labelmode = labelmode
        # self.chunks = int(chunks)
        self.multimode = multimode

        self._N_VIDEO_FRAMES = chunks

        self.mono = mono

        self.predict_flag = predict_flag
        self.load_frame = LoadVideoFrame(self._N_VIDEO_FRAMES,
                                         self.vidreaders,
                                         self.camnames,
                                         self.predict_flag)

    def __len__(self):
        """Denote the number of batches per epoch."""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
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

    def load_tif_frame(self, ind, camname):
        """Load frames in tif mode."""
        # In tif mode, vidreaders should just be paths to the tif directory
        return imageio.imread(
            os.path.join(self.vidreaders[camname], "{}.tif".format(ind))
        )

    def __data_generation(self, list_IDs_temp):
        """Generate data containing batch_size samples.

        # X : (n_samples, *dim, n_channels)
        """
        # Initialization
        X = np.empty(
            (
                self.batch_size * len(self.camnames[0]),
                *self.dim_in,
                self.n_channels_in,
            ),
            dtype="uint8",
        )

        # We'll need to transpose this later such that channels are last,
        # but initializaing the array this ways gives us
        # more flexibility in terms of user-defined array sizes\
        if self.labelmode == "prob":
            y = np.empty(
                (
                    self.batch_size * len(self.camnames[0]),
                    self.n_channels_out,
                    *self.dim_out,
                ),
                dtype="float32",
            )
        else:
            # Just return the targets, without making a meshgrid later
            y = np.empty(
                (
                    self.batch_size * len(self.camnames[0]),
                    self.n_channels_out,
                    len(self.dim_out),
                ),
                dtype="float32",
            )

        # Generate data
        cnt = 0
        for i, ID in enumerate(list_IDs_temp):
            if "_" in ID:
                experimentID = int(ID.split("_")[0])
            else:
                # Then we only have one experiment
                experimentID = 0
            for camname in self.camnames[experimentID]:
                # Store sample
                # TODO(Refactor): This section is tricky to read
                if self.immode == "video":
                    X[cnt] = self.load_frame.load_vid_frame(
                        self.labels[ID]["frames"][camname],
                        camname,
                        self.extension,
                    )[
                        self.crop_height[0] : self.crop_height[1],
                        self.crop_width[0] : self.crop_width[1],
                    ]
                elif self.immode == "tif":
                    X[cnt] = self.load_tif_frame(
                        self.labels[ID]["frames"][camname], camname
                    )[
                        self.crop_height[0] : self.crop_height[1],
                        self.crop_width[0] : self.crop_width[1],
                    ]
                else:
                    raise Exception("Not a valid image reading mode")

                # Labels will now be the pixel positions of each joint.
                # Here, we convert them to
                # probability maps with a numpy meshgrid operation
                this_y = np.round(self.labels[ID]["data"][camname])
                if self.immode == "video":
                    this_y[0, :] = this_y[0, :] - self.crop_width[0]
                    this_y[1, :] = this_y[1, :] - self.crop_height[0]
                else:
                    raise Exception(
                        "Unsupported image format. Needs to be video files."
                    )
                # import pdb
                # pdb.set_trace()
                if self.labelmode == "prob":
                    (x_coord, y_coord) = np.meshgrid(
                        np.arange(self.dim_out[1]), np.arange(self.dim_out[0])
                    )

                    # Get the probability maps for all instances
                    instance_prob = []
                    for instance in range(self.n_instances):
                        instance_prob.append(
                            np.exp(
                                -(
                                    (y_coord - this_y[1, instance]) ** 2
                                    + (x_coord - this_y[0, instance]) ** 2
                                )
                                / (2 * self.out_scale ** 2)
                            )
                        )

                    # If using single-channel multi_instance take the max
                    # across probability maps. Otherwise assign a probability
                    # map to each channel.
                    if self.n_channels_out == 1:
                        y[cnt, 0] = np.max(np.stack(instance_prob, axis=2), axis=2)
                    else:
                        if len(instance_prob) != self.n_channels_out:
                            raise ValueError(
                                "n_channels_out != n_instances. This is necessary for multi-channel multi-instance tracking."
                            )
                        for j, instance in enumerate(instance_prob):
                            y[cnt, j] = instance
                else:
                    y[cnt] = this_y.T
                # plt.imshow(np.squeeze(y[0,:,:,:]))
                # plt.show()
                # import pdb
                # pdb.set_trace()
                cnt = cnt + 1

        # Move channels last
        if self.labelmode == "prob":
            y = np.transpose(y, [0, 2, 3, 1])
        else:
            # One less dimension when not training with probability map targets
            y = np.transpose(y, [0, 2, 1])

        if self.downsample > 1:
            X = processing.downsample_batch(X, fac=self.downsample, method=self.dsmode)
            if self.labelmode == "prob":
                y = processing.downsample_batch(
                    y, fac=self.downsample, method=self.dsmode
                )
                y /= np.max(np.max(y, axis=1), axis=1)[:, np.newaxis, np.newaxis, :]

        if self.mono and self.n_channels_in == 3:
            # Go from 3 to 1 channel using RGB conversion. This will also
            # work fine if there are just 3 channel grayscale
            X = X[:, :, :, 0] * 0.2125 + X[:, :, :, 1] * 0.7154 + X[:, :, :, 2] * 0.0721

            X = X[:, :, :, np.newaxis]

        if self.mono:
            # Just subtract the mean imagent BGR value, which is as close as we
            # get to vgg19 normalization
            X -= 114.67
        else:
            X = pp_vgg19(X)
        return X, y


class DataGenerator_downsample_frommem(keras.utils.Sequence):
    """Generate 3d conv data from memory."""

    def __init__(
        self,
        list_IDs,
        data,
        labels,
        batch_size,
        chan_num=3,
        shuffle=True,
        augment_brightness=False,
        augment_hue=False,
        augment_rotation=False,
        augment_zoom=False,
        augment_shear=False,
        augment_shift=False,
        bright_val=0.05,
        hue_val=0.05,
        shift_val=0.05,
        rotation_val=5,
        shear_val=5,
        zoom_val=0.05,
    ):
        """Initialize data generator."""
        self.list_IDs = list_IDs
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.chan_num = chan_num
        self.shuffle = shuffle

        self.augment_brightness = augment_brightness
        self.augment_hue = augment_hue
        self.augment_rotation = augment_rotation
        self.augment_zoom = augment_zoom
        self.augment_shear = augment_shear
        self.augment_shift = augment_shift
        self.bright_val = bright_val
        self.hue_val = hue_val
        self.shift_val = shift_val
        self.rotation_val = rotation_val
        self.shear_val = shear_val
        self.zoom_val = zoom_val
        self.on_epoch_end()

    def __len__(self):
        """Denote the number of batches per epoch."""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        """Update indexes after each epoch."""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def shift_im(self, im, lim, dim=2):
        ulim = im.shape[dim] - np.abs(lim)

        if dim == 2:
            if lim < 0:
                im[:, :, :ulim] = im[:, :, np.abs(lim) :]
                im[:, :, ulim:] = im[:, :, ulim : ulim + 1]
            else:
                im[:, :, lim:] = im[:, :, :ulim]
                im[:, :, :lim] = im[:, :, lim : lim + 1]
        elif dim == 1:
            if lim < 0:
                im[:, :ulim] = im[:, np.abs(lim) :]
                im[:, ulim:] = im[:, ulim : ulim + 1]
            else:
                im[:, lim:] = im[:, :ulim]
                im[:, :lim] = im[:, lim : lim + 1]
        else:
            raise Exception("Not a valid dimension for shift indexing")

        return im

    def random_shift(self, X, y_2d, im_h, im_w, scale):
        """
        Randomly shifts all images in batch, in the range [-im_w*scale, im_w*scale]
            and [im_h*scale, im_h*scale]
        """
        wrng = np.random.randint(-int(im_w * scale), int(im_w * scale))
        hrng = np.random.randint(-int(im_h * scale), int(im_h * scale))

        X = self.shift_im(X, wrng)
        X = self.shift_im(X, hrng, dim=1)

        y_2d = self.shift_im(y_2d, wrng)
        y_2d = self.shift_im(y_2d, hrng, dim=1)

        return X, y_2d

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def __data_generation(self, list_IDs_temp):
        """Generate data containing batch_size samples."""
        # Initialization

        X = np.zeros((self.batch_size, *self.data.shape[1:]))
        y_2d = np.zeros((self.batch_size, *self.labels.shape[1:]))

        for i, ID in enumerate(list_IDs_temp):
            X[i] = self.data[ID].copy()
            y_2d[i] = self.labels[ID]

        if self.augment_rotation or self.augment_shear or self.augment_zoom:

            affine = {}
            affine["zoom"] = 1
            affine["rotation"] = 0
            affine["shear"] = 0

            # Because we use views down below,
            # don't change the targets in memory.
            # But also, don't deep copy y_2d unless necessary (that's
            # why it's here and not above)
            y_2d = y_2d.copy()

            if self.augment_rotation:
                affine["rotation"] = self.rotation_val * (np.random.rand() * 2 - 1)
            if self.augment_zoom:
                affine["zoom"] = self.zoom_val * (np.random.rand() * 2 - 1) + 1
            if self.augment_shear:
                affine["shear"] = self.shear_val * (np.random.rand() * 2 - 1)

            for idx in range(X.shape[0]):
                X[idx] = tf.keras.preprocessing.image.apply_affine_transform(
                    X[idx],
                    theta=affine["rotation"],
                    shear=affine["shear"],
                    zx=affine["zoom"],
                    zy=affine["zoom"],
                    fill_mode="nearest",
                )
                y_2d[idx] = tf.keras.preprocessing.image.apply_affine_transform(
                    y_2d[idx],
                    theta=affine["rotation"],
                    shear=affine["shear"],
                    zx=affine["zoom"],
                    zy=affine["zoom"],
                    fill_mode="nearest",
                )

        if self.augment_shift:
            X, y_2d = self.random_shift(
                X, y_2d.copy(), X.shape[1], X.shape[2], self.shift_val
            )

        if self.augment_brightness:
            X = tf.image.random_brightness(X, self.bright_val)

        if self.augment_hue:
            if self.chan_num == 3:
                X = tf.image.random_hue(X, self.hue_val)
            else:
                warnings.warn("Hue augmention set to True for mono. Ignoring.")

        if self.augment_brightness or self.augment_hue:
            X = X.numpy()

        return X, y_2d

    def save_for_dlc(self, imfolder, ext=".png", full_data=True, compress_level=9):
        """Generate data.

        # The full_data flag is used so that one can
        # write only the coordinates and not the images, if desired.
        """
        # We don't allow for multiple experiments here
        cnt = 0
        self.camnames = self.camnames[0]
        warnings.warn(
            "Note: generate_labels does not  \
            support multiple experiments at once. Converting camnames from dict to list"
        )
        list_IDs_temp = self.list_IDs
        dsize = self.labels[list_IDs_temp[0]]["data"][self.camnames[0]].shape
        allcoords = np.zeros(
            (len(list_IDs_temp) * len(self.camnames), dsize[1], 3), dtype="int"
        )
        fnames = []

        # Load in a sample so that size can be found when full_data=False
        camname = self.camnames[0]
        # TODO(refactor): Hard to read
        X = self.load_frame.load_vid_frame(
            self.labels[list_IDs_temp[0]]["frames"][camname],
            camname,
            self.extension,
        )[
            self.crop_height[0] : self.crop_height[1],
            self.crop_width[0] : self.crop_width[1],
        ]

        for i, ID in enumerate(list_IDs_temp):
            for camname in self.camnames:
                if full_data:
                    X = self.load_frame.load_vid_frame(
                        self.labels[ID]["frames"][camname],
                        camname,
                        self.extension,
                    )[
                        self.crop_height[0] : self.crop_height[1],
                        self.crop_width[0] : self.crop_width[1],
                    ]

                # Labels will now be the pixel positions of each joint.
                # Here, we convert them to probability maps with a numpy
                # meshgrid operation
                this_y = self.labels[ID]["data"][camname].copy()
                this_y[0, :] = this_y[0, :] - self.crop_width[0]
                this_y[1, :] = this_y[1, :] - self.crop_height[0]

                if self.downsample > 1:
                    X = processing.downsample_batch(
                        X[np.newaxis, :, :, :],
                        fac=self.downsample,
                        method="dsm",
                    )
                    this_y = np.round(this_y / 2).astype("int")
                    if full_data:
                        imageio.imwrite(
                            imfolder + "sample{}_".format(ID) + camname + ext,
                            X[0].astype("uint8"),
                            compress_level=compress_level,
                        )
                else:
                    if full_data:
                        imageio.imwrite(
                            imfolder + "sample{}_".format(ID) + camname + ext,
                            X.astype("uint8"),
                            compress_level=compress_level,
                        )

                allcoords[cnt, :, 0] = np.arange(dsize[1])
                allcoords[cnt, :, 1:] = this_y.T

                # TODO(os.path): This is unix-specific
                # These paths should be using AWS/UNIX only
                relpath = imfolder.split(os.sep)[-2]
                relpath = (
                    ".."
                    + os.sep
                    + relpath
                    + os.sep
                    + "sample{}_".format(ID)
                    + camname
                    + ext
                )
                fnames.append(relpath)

                cnt = cnt + 1

        sio.savemat(
            imfolder + "allcoords.mat",
            {
                "allcoords": allcoords,
                "imsize": [X.shape[-1], X.shape[0], X.shape[1]],
                "filenames": fnames,
            },
        )
