"""Setup file for dannce."""
from setuptools import setup

setup(
    name='dannce',
    version='1.0.0',
    install_requires=['numpy',
                      'scipy',
                      'six',
                      'pyyaml',
                      'imageio==2.8.0',
                      'imageio-ffmpeg',
                      'scikit-image',
                      'matplotlib',
                      'opencv-python',
                      'tensorflow',
                      'torch']
)
