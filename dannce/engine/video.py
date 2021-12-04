""" Video reading and writing interfaces for different formats. """
from copy import Error
import os
import cv2
import numpy as np
import attr
import multiprocessing
import imageio
import time
from typing import List, Dict, Tuple, Text


@attr.s(auto_attribs=True, eq=False, order=False)
class MediaVideo:
    """
    Video data stored in traditional media formats readable by FFMPEG
    This class provides bare minimum read only interface on top of
    OpenCV's VideoCapture class.
    Args:
        filename: The name of the file (.mp4, .avi, etc)
        grayscale: Whether the video is grayscale or not. "auto" means detect
            based on first frame.
        bgr: Whether color channels ordered as (blue, green, red).
    """

    filename: str = attr.ib()
    grayscale: bool = attr.ib()
    bgr: bool = attr.ib(default=True)

    # Unused attributes still here so we don't break deserialization
    dataset: str = attr.ib(default="")
    input_format: str = attr.ib(default="")

    _detect_grayscale = False
    _reader_ = None
    _test_frame_ = None

    @property
    def __lock(self):
        if not hasattr(self, "_lock"):
            self._lock = multiprocessing.RLock()
        return self._lock

    @grayscale.default
    def __grayscale_default__(self):
        self._detect_grayscale = True
        return False

    @property
    def __reader(self):
        # Load if not already loaded
        if self._reader_ is None:
            if not os.path.isfile(self.filename):
                raise FileNotFoundError(
                    f"Could not find filename video filename named {self.filename}"
                )

            # Try and open the file either locally in current directory or with full
            # path
            self._reader_ = cv2.VideoCapture(self.filename)

            # If the user specified None for grayscale bool, figure it out based on the
            # the first frame of data.
            if self._detect_grayscale is True:
                self.grayscale = bool(
                    np.alltrue(self.test_frame[..., 0] == self.test_frame[..., -1])
                )

        # Return cached reader
        return self._reader_

    @property
    def __frames_float(self):
        return self.__reader.get(cv2.CAP_PROP_FRAME_COUNT)

    @property
    def test_frame(self):
        # Load if not already loaded
        if self._test_frame_ is None:
            # Lets grab a test frame to help us figure things out about the video
            self._test_frame_ = self.get_frame(0, grayscale=False)

        # Return stored test frame
        return self._test_frame_

    def matches(self, other: "MediaVideo") -> bool:
        """
        Check if attributes match those of another video.
        Args:
            other: The other video to compare with.
        Returns:
            True if attributes match, False otherwise.
        """
        return (
            self.filename == other.filename
            and self.grayscale == other.grayscale
            and self.bgr == other.bgr
        )

    @property
    def fps(self) -> float:
        """Returns frames per second of video."""
        return self.__reader.get(cv2.CAP_PROP_FPS)

    # The properties and methods below complete our contract with the
    # higher level Video interface.

    @property
    def frames(self):
        """See :class:`Video`."""
        return int(self.__frames_float)

    @property
    def channels(self):
        """See :class:`Video`."""
        if self.grayscale:
            return 1
        else:
            return self.test_frame.shape[2]

    @property
    def width(self):
        """See :class:`Video`."""
        return self.test_frame.shape[1]

    @property
    def height(self):
        """See :class:`Video`."""
        return self.test_frame.shape[0]

    @property
    def dtype(self):
        """See :class:`Video`."""
        return self.test_frame.dtype

    def reset(self):
        """Reloads the video."""
        self._reader_ = None

    def get_frame(self, idx: int, grayscale: bool = None) -> np.ndarray:
        """See :class:`Video`."""

        with self.__lock:
            if self.__reader.get(cv2.CAP_PROP_POS_FRAMES) != idx:
                self.__reader.set(cv2.CAP_PROP_POS_FRAMES, idx)

            success, frame = self.__reader.read()

        if not success or frame is None:
            raise KeyError(f"Unable to load frame {idx} from {self}.")

        if grayscale is None:
            grayscale = self.grayscale

        if grayscale:
            frame = frame[..., 0][..., None]

        if self.bgr:
            frame = frame[..., ::-1]

        return frame


class LoadVideoFrame:
    """
    This class generalized load_vid_frame for access by all generators
    Args:
        _N_VIDEO_FRAMES: Array of chunked video indices
        vidreaders: Dictionary of all video file paths
        camnames: All Camera names
        predict_flag: If True, uses imageio rather than OpenCV
    """

    def __init__(self, _N_VIDEO_FRAMES, vidreaders, camnames, predict_flag):

        self._N_VIDEO_FRAMES = _N_VIDEO_FRAMES
        self.vidreaders = vidreaders
        self.camnames = camnames
        self.predict_flag = predict_flag

        # we keep a running video object so at least we don't open a new one every time
        self.currvideo = {}
        self.currvideo_name = {}
        for dd in camnames.keys():
            for cc in camnames[dd]:
                self.currvideo[cc] = None
                self.currvideo_name[cc] = None

    def load_vid_frame(
        self, ind: int, camname: Text, extension: Text = ".mp4"
    ) -> np.ndarray:
        """Load video frame from a single camera.

        This is currently implemented for handling only one camera as input

        Args:
            ind (int): Frame index
            camname (Text): Camera index
            extension (Text, optional): Video extension

        Returns:
            np.ndarray: Video frame as w x h x c numpy ndarray
        """
        chunks = self._N_VIDEO_FRAMES[camname]
        cur_video_id = np.nonzero([c <= ind for c in chunks])[0][-1]
        cur_first_frame = chunks[cur_video_id]
        fname = str(cur_first_frame) + extension
        frame_num = int(ind - cur_first_frame)

        keyname = os.path.join(camname, fname)

        thisvid_name = self.vidreaders[camname][keyname]
        abname = thisvid_name.split("/")[-1]
        if abname == self.currvideo_name[camname]:
            vid = self.currvideo[camname]
        else:
            # use imageio for prediction, because linear seeking
            # is faster with imageio than opencv
            try:
                vid = (
                    imageio.get_reader(thisvid_name)
                    if self.predict_flag
                    else MediaVideo(thisvid_name, grayscale=False)
                )
            except (OSError, IOError, RuntimeError):
                time.sleep(2)
                vid = (
                    imageio.get_reader(thisvid_name)
                    if self.predict_flag
                    else MediaVideo(thisvid_name, grayscale=False)
                )
            print("Loading new video: {} for {}".format(abname, camname))
            self.currvideo_name[camname] = abname
            # close current vid
            # Without a sleep here, ffmpeg can hang on video close
            time.sleep(0.25)

            # Close previously opened and unneeded videos by their camera name
            # Assumes the camera names do not contain underscores other than the expid. 
            # previous_camera_name = "_".join(camname.split("_")[1:])
            previous_camera_name = camname.split("_")[-1]
            for key, value in self.currvideo.items():
                if previous_camera_name in key:
                    if value is not None:
                        self.currvideo[
                            key
                        ].close() if self.predict_flag else self.currvideo[
                            key
                        ]._reader_.release()
            self.currvideo[camname] = vid
        im = self._load_frame_multiple_attempts(frame_num, vid)
        return im

    def _load_frame_multiple_attempts(self, frame_num, vid, n_attempts=10):
        attempts = 0
        while attempts < n_attempts:
            im = self._load_frame(frame_num, vid)
            if im is None:
                attempts += 1
            else:
                break
        else:
            raise KeyError
        return im

    def _load_frame(self, frame_num, vid):
        im = None
        try:
            im = (
                vid.get_data(frame_num).astype("uint8")
                if self.predict_flag
                else vid.get_frame(frame_num)
            )
        # This deals with a strange indexing error in the pup data.
        except IndexError:
            print("Indexing error, using previous frame")
            im = (
                vid.get_data(frame_num - 1).astype("uint8")
                if self.predict_flag
                else vid.get_frame(frame_num - 1)
            )
        # Files can lock if other processes are also trying to access the data.
        except KeyError:
            time.sleep(5)
            pass
        return im
