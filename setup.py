"""Setup file for dannce."""
from setuptools import setup

setup(
    name='dannce',
    version='0.0.0',
    install_requires=['numpy',
                      'scipy',
                      'six',
                      'keras==2.2.2',
                      'imageio',
                      'scikit-image',
                      'matlab',
                      'matplotlib']
)
