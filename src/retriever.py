"""
retriever: this abstracts out the various different data retrieval methods available
"""
from abc import ABC, abstractmethod
import json
import subprocess
import ffmpeg
import numpy as np

class AbstractRetriever(ABC):
    """Docstring for AbstractRetriever. """

    @abstractmethod
    def retrieve(self):
        """returns an iterator of 3d numpy arrays that act as video"""
        pass

    @abstractmethod
    def reset(self):
        """resets the retriever"""
        pass

class MPEGRetriever(AbstractRetriever):
    """Docstring for MPEGRetriever. """

    def __init__(self, file):
        self.file = file
        self.width, self.height = None, None

    def _get_metadata(self):
        ffprobe_call = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            self.file
        ]
        info = json.load(subprocess.Popen(ffprobe_call, stdout=subprocess.PIPE).stdout)
        self.width, self.height = info['streams'][0]['width'], info['streams'][0]['height']

    def retrieve(self):
        if self.width is None or self.height is None:
            self._get_metadata()

        out, _ = (
            ffmpeg
            .input(self.file)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, capture_stderr=True)
        )

        return (
            np
            .frombuffer(out, np.uint8)
            .reshape([-1, self.width, self.height, 3])
        )

    @staticmethod
    def reset():
        pass
