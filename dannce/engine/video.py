""" Video reading and writing interfaces for different formats. """
import os
import cv2
import numpy as np
import attr
import multiprocessing


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