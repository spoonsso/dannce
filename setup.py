"""Setup file for dannce."""
from setuptools import setup

setup(
    name='dannce',
    version='0.0.1',
    install_requires=['numpy',
                      'scipy',
                      'six',
                      'keras==2.2.4',
                      'imageio',
                      'imageio-ffmpeg',
                      'scikit-image',
                      'matplotlib',
                      'opencv-python']
)
