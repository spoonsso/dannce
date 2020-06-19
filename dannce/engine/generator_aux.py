"""Generator for 3d video images."""
import numpy as np
import tensorflow.keras as keras
import os
from tensorflow.keras.applications.vgg19 import preprocess_input as pp_vgg19
import imageio
from dannce.engine import processing as processing
import scipy.io as sio
import warnings
import time

_DEFAULT_CAM_NAMES = ["CameraR", "CameraL", "CameraU", "CameraU2", "CameraS", "CameraE"]
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
        preload=True,
        dsmode="dsm",
        chunks=3500,
        multimode=False,
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
        self.preload = preload
        self.dsmode = dsmode
        self.on_epoch_end()

        if immode == "video":
            self.extension = (
                "." + list(vidreaders[camnames[0][0]].keys())[0].rsplit(".")[-1]
            )

        self.immode = immode
        self.labelmode = labelmode
        self.chunks = int(chunks)
        self.multimode = multimode

        self._N_VIDEO_FRAMES = self.chunks

        if not self.preload:
            # then we keep a running video object so at least we don't open a new one every time
            self.currvideo = {}
            self.currvideo_name = {}
            for dd in camnames.keys():
                for cc in camnames[dd]:
                    self.currvideo[cc] = None
                    self.currvideo_name[cc] = None

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

    def load_vid_frame(self, ind, camname, preload=True, extension=".mp4"):
        """Load the video frame from a single camera."""
        fname = (
            str(self._N_VIDEO_FRAMES * int(np.floor(ind / self._N_VIDEO_FRAMES)))
            + extension
        )
        frame_num = int(ind % self._N_VIDEO_FRAMES)
        keyname = os.path.join(camname, fname)
        if preload:
            return self.vidreaders[camname][keyname].get_data(frame_num)
        else:
            thisvid_name = self.vidreaders[camname][keyname]
            abname = thisvid_name.split("/")[-1]
            if abname == self.currvideo_name[camname]:
                vid = self.currvideo[camname]
            else:
                vid = imageio.get_reader(thisvid_name)
                print("Loading new video: {} for {}".format(abname, camname))
                self.currvideo_name[camname] = abname
                # close current vid
                # Without a sleep here, ffmpeg can hang on video close
                time.sleep(0.25)
                if self.currvideo[camname] is not None:
                    self.currvideo[camname].close()
                self.currvideo[camname] = vid

            im = vid.get_data(frame_num)

            return im

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
            for camname in self.camnames[experimentID]:
                # Store sample
                # TODO(Refactor): This section is tricky to read
                if self.immode == "video":
                    X[cnt] = self.load_vid_frame(
                        self.labels[ID]["frames"][camname],
                        camname,
                        self.preload,
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

        return pp_vgg19(X), y

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
        X = self.load_vid_frame(
            self.labels[list_IDs_temp[0]]["frames"][camname],
            camname,
            self.preload,
            self.extension,
        )[
            self.crop_height[0] : self.crop_height[1],
            self.crop_width[0] : self.crop_width[1],
        ]

        for i, ID in enumerate(list_IDs_temp):
            for camname in self.camnames:
                if full_data:
                    X = self.load_vid_frame(
                        self.labels[ID]["frames"][camname],
                        camname,
                        self.preload,
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
                        X[np.newaxis, :, :, :], fac=self.downsample, method="dsm"
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
