"""Setup file for daance."""
from setuptools import setup

setup(
    name='daance',
    version='0.0.0',
    install_requires=['numpy',
                      'scipy',
                      'six',
                      'keras==2.2.2',
                      'imageio',
                      'scikit-image',
                      'matplotlib']
)
