"""
This script should be used to test all the additional features that are implemented into the model.
For example, new loss functions, metrics, normalizations and regularization techniques.

"""

from absl.testing import absltest
import tensorflow as tf
import dannce.cli as cli
import os
import numpy as np
import scipy.io as sio
import sys
import unittest
from unittest.mock import patch
from typing import Text
import datetime
from cli_test import *


# Initialize the gpu prior to testing
# tf.test.is_gpu_available()
tf.config.list_physical_devices('GPU')

# Move to the testing project folder
os.chdir("configs")



class TestDannceLosses(absltest.TestCase):
    def test_dannce_train_finetune_max(self, add_args):
        train_setup()
        args = [
            "dannce-train",
            "config_mousetest.yaml",
            "--net-type=MAX",
            "--dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/weights/weights.rat.MAX/",
        ]

        if add_args is not None:
            args = args + add_args
            
        with patch("sys.argv", args):
            cli.dannce_train_cli()

    def test_dannce_train_finetune_avg(self):
        train_setup()
        args = [
            "dannce-train",
            "config_mousetest.yaml",
            "--net-type=AVG",
            "--loss=log_cosh_loss",
            "--dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/weights/",
        ]
        with patch("sys.argv", args):
            cli.dannce_train_cli()

    def test_dannce_train_finetune_avg_heatmap_regularization(self):
        train_setup()
        args = [
            "dannce-train",
            "config_mousetest.yaml",
            "--net-type=AVG",
            "--loss=log_cosh_loss",
            "--dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/weights/",
        ]
        with patch("sys.argv", args):
            cli.dannce_train_cli()

    def test_dannce_train_finetune_avg_from_finetune(self):
        train_setup()
        args = [
            "dannce-train",
            "config_mousetest.yaml",
            "--net-type=AVG",
            "--loss=log_cosh_loss",
            "--dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/train_results/AVG/",
        ]
        with patch("sys.argv", args):
            cli.dannce_train_cli()

    def test_dannce_train_avg(self):
        train_setup()
        args = [
            "dannce-train",
            "config_mousetest.yaml",
            "--net=unet3d_big_expectedvalue",
            "--train-mode=new",
            "--loss=log_cosh_loss",
            "--n-channels-out=22",
        ]
        with patch("sys.argv", args):
            cli.dannce_train_cli()

    def test_dannce_train_max(self):
        train_setup()
        args = [
            "dannce-train",
            "config_mousetest.yaml",
            "--net=unet3d_big",
            "--train-mode=new",
            "--loss=log_cosh_loss",
            "--n-channels-out=22",
        ]
        with patch("sys.argv", args):
            cli.dannce_train_cli()

    def test_dannce_train_avg_continued(self):
        train_setup()
        args = [
            "dannce-train",
            "config_mousetest.yaml",
            "--net-type=AVG",
            "--train-mode=continued",
            "--loss=log_cosh_loss",
            "--dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/train_results/AVG/",
        ]
        with patch("sys.argv", args):
            cli.dannce_train_cli()

    def test_dannce_train_max_continued(self):
        train_setup()
        args = [
            "dannce-train",
            "config_mousetest.yaml",
            "--net=finetune_MAX",
            "--train-mode=continued",
            "--loss=log_cosh_loss",
            "--dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/train_results/",
        ]
        with patch("sys.argv", args):
            cli.dannce_train_cli()

    def test_dannce_train_avg_mono(self):
        train_setup()
        args = [
            "dannce-train",
            "config_mousetest.yaml",
            "--net-type=AVG",
            "--train-mode=new",
            "--net=unet3d_big_expectedvalue",
            "--mono=True",
            "--loss=log_cosh_loss",
            "--n-channels-out=22",
        ]
        with patch("sys.argv", args):
            cli.dannce_train_cli()

    def test_dannce_train_avg_mono_finetune(self):
        train_setup()
        args = [
            "dannce-train",
            "config_mousetest.yaml",
            "--net-type=AVG",
            "--mono=True",
            "--loss=log_cosh_loss",
            "--dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/weights/weights.rat.AVG.MONO/",
        ]
        with patch("sys.argv", args):
            cli.dannce_train_cli()

    def test_dannce_train_avg_mono_finetune_drop_landmarks(self):
        train_setup()
        args = [
            "dannce-train",
            "config_mousetest.yaml",
            "--net-type=AVG",
            "--mono=True",
            "--loss=log_cosh_loss",
            "--dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/weights/weights.rat.AVG.MONO/",
            "--drop-landmark=[5,7]",
        ]
        with patch("sys.argv", args):
            cli.dannce_train_cli()

    def test_npy_volume_with_validation(self):
        os.chdir("../../demo/markerless_mouse_1/")
        args = [
            "dannce-train",
            "../../configs/dannce_mouse_config.yaml",
            "--net-type=AVG",
            "--use-npy=True",
            "--dannce-train-dir=./DANNCE/npy_test/",
            "--epochs=2",
            "--valid-exp=[1]",
            "--gpu=1",
        ]
        with patch("sys.argv", args):
            cli.dannce_train_cli()

    def test_npy_volume_with_multi_gpu(self):
        os.chdir("../../demo/markerless_mouse_1/")
        args = [
            "dannce-train",
            "../../configs/dannce_mouse_config.yaml",
            "--net-type=AVG",
            "--batch-size=4",
            "--use-npy=True",
            "--dannce-train-dir=./DANNCE/npy_test/",
            "--epochs=2",
            "--valid-exp=[1]",
            "--multi-gpu-train=True",
            "--gpu=1",
        ]
        with patch("sys.argv", args):
            cli.dannce_train_cli()

    def test_npy_volume_with_num_train_exp(self):
        os.chdir("../../demo/markerless_mouse_1/")
        args = [
            "dannce-train",
            "../../configs/dannce_mouse_config.yaml",
            "--net-type=AVG",
            "--use-npy=True",
            "--dannce-train-dir=./DANNCE/npy_test/",
            "--epochs=2",
            "--num-train-per-exp=2",
            "--batch-size=1",
            "--gpu=1",
        ]
        with patch("sys.argv", args):
            cli.dannce_train_cli()

    def test_npy_volume_with_validation_and_num_train_exp(self):
        os.chdir("../../demo/markerless_mouse_1/")
        args = [
            "dannce-train",
            "../../configs/dannce_mouse_config.yaml",
            "--net-type=AVG",
            "--use-npy=True",
            "--dannce-train-dir=./DANNCE/npy_test/",
            "--epochs=2",
            "--valid-exp=[1]",
            "--num-train-per-exp=2",
            "--batch-size=1",
            "--gpu=1",
        ]
        with patch("sys.argv", args):
            cli.dannce_train_cli()

    def test_dannce_train_MAX_layer_norm(self):
        train_setup()
        args = [
            "dannce-train",
            "dgptest_config.yaml",
            "--dannce-train-dir=../../demo/markerless_mouse_1/DANNCE/train_test_ln/",
            "--n-channels-out=22",
        ]
        with patch("sys.argv", args):
            cli.dannce_train_cli()

    def test_dannce_train_MAX_scratch_instance_norm(self):
        train_setup()
        args = [
            "dannce-train",
            "dgptest_config.yaml",
            "--norm-method=instance",
            "--dannce-train-dir=../../demo/markerless_mouse_1/DANNCE/train_test_in/",
            "--n-channels-out=22",
        ]
        with patch("sys.argv", args):
            cli.dannce_train_cli()

    def test_dannce_train_DGP_MAX_scratch_layer_norm_sigmoid_cross_entropy_Gaussian(
        self,
    ):
        train_setup()
        args = [
            "dannce-train",
            "dgptest_config.yaml",
            "--loss=gaussian_cross_entropy_loss",
            "--dannce-train-dir=../../demo/markerless_mouse_1/DANNCE/train_test_ln_dgp/",
            "--n-channels-out=22",
        ]
        with patch("sys.argv", args):
            cli.dannce_train_cli()

    def test_dannce_train_DGP_MAX_scratch_instance_norm_sigmoid_cross_entropy_Gaussian(
        self,
    ):
        train_setup()
        args = [
            "dannce-train",
            "dgptest_config.yaml",
            "--norm-method=instance",
            "--loss=gaussian_cross_entropy_loss",
            "--dannce-train-dir=../../demo/markerless_mouse_1/DANNCE/train_test_ln_dgp/",
            "--n-channels-out=22",
        ]
        with patch("sys.argv", args):
            cli.dannce_train_cli()


class TestDanncePredict(absltest.TestCase):
    def test_dannce_predict_mono(self):
        # TODO(refactor): This test depends on there being a mono model saved.
        train_setup()
        args = [
            "dannce-train",
            "config_mousetest.yaml",
            "--net-type=AVG",
            "--mono=True",
            "--dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/weights/weights.rat.AVG.MONO/",
        ]
        with patch("sys.argv", args):
            cli.dannce_train_cli()
        args = [
            "dannce-predict",
            "config_mousetest.yaml",
            "--net-type=AVG",
            "--dannce-predict-model=../../demo/markerless_mouse_1/DANNCE/train_test/fullmodel_weights/fullmodel_end.hdf5",
            "--mono=True",
        ]
        with patch("sys.argv", args):
            cli.dannce_predict_cli()

    def test_dannce_predict_avg(self):
        train_setup()
        args = [
            "dannce-predict",
            "config_mousetest.yaml",
            "--net-type=AVG",
        ]
        with patch("sys.argv", args):
            cli.dannce_predict_cli()
        self.assertTrue(compare_predictions(
            "../touchstones/save_data_AVG_torch_nearest.mat",
            "../../demo/markerless_mouse_1/DANNCE/predict_test/save_data_AVG0.mat",
        ))

    def test_dannce_predict_max(self):
        train_setup()
        args = [
            "dannce-predict",
            "config_mousetest.yaml",
            "--net-type=MAX",
            "--expval=False",
            "--dannce-predict-model=../../demo/markerless_mouse_1/DANNCE/train_results/weights.12000-0.00014.hdf5",
        ]
        with patch("sys.argv", args):
            cli.dannce_predict_cli()
        self.assertTrue(compare_predictions(
            "../touchstones/save_data_MAX_torchnearest_newtfroutine.mat",
            "../../demo/markerless_mouse_1/DANNCE/predict_test/save_data_MAX0.mat",
        ))

    def test_dannce_predict_numpy_volume_generation(self):
        setup = "cp ./label3d_voltest_dannce_m1.mat ./alabel3d_temp_dannce.mat"
        os.system(setup)
        args = [
            "dannce-predict",
            "config_mousetest.yaml",
            "--net-type=AVG",
            "--write-npy=../../demo/markerless_mouse_1/npy_volumes/",
            "--batch-size=1",
        ]
        with patch("sys.argv", args):
            cli.dannce_predict_cli()
        setup2 = "cp ./label3d_voltest_dannce_m2.mat ./alabel3d_temp_dannce.mat"
        os.system(setup2)
        args = [
            "dannce-predict",
            "config_mousetest.yaml",
            "--net-type=AVG",
            "--write-npy=../../demo/markerless_mouse_2/npy_volumes/",
            "--batch-size=1",
        ]
        with patch("sys.argv", args):
            cli.dannce_predict_cli()


class myTextTestResult(unittest.TextTestResult):
    def startTest(self, test: unittest.case.TestCase) -> None:
        self.stream.write("\n")
        self.stream.write(str(datetime.datetime.now()) + ": ")
        return super(myTextTestResult, self).startTest(test)

if __name__ == "__main__":
    log_file = 'log_file.txt'
    with open(log_file, "w") as f:
       runner = unittest.TextTestRunner(f, resultclass=myTextTestResult)
       absltest.main(testRunner=runner)
