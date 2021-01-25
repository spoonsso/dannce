"""Setup file for dannce."""
from setuptools import setup, find_packages

setup(
    name="dannce",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "six",
        "pyyaml",
        "imageio==2.8.0",
        "imageio-ffmpeg",
        "scikit-image",
        "matplotlib",
        "opencv-python",
<<<<<<< HEAD
        "tensorflow==2.3.1",
=======
        "tensorflow==2.3.0",
>>>>>>> sampleID_fix
        "torch",
        "numpy==1.18"
    ],
    entry_points={
        "console_scripts": [
            "dannce-train = dannce.cli:dannce_train_cli",
            "dannce-predict = dannce.cli:dannce_predict_cli",
            "com-train = dannce.cli:com_train_cli",
            "com-predict = dannce.cli:com_predict_cli",
        ]
    },
)
