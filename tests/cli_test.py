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

# Initialize the gpu prior to testing
# tf.test.is_gpu_available()
tf.config.list_physical_devices('GPU')

# Move to the testing project folder
os.chdir("configs")
N_TEST_IMAGES = 8

def compare_predictions(file_1: Text, file_2: Text, th: float = 0.05):
    """Compares two prediction matfiles.

    Args:
        file_1 (Text): Path to prediction file 1
        file_2 (Text): Path to prediction file 2
        th (float, optional): Testing tolerance in mm. Defaults to 0.05.

    Raises:
        Exception: If file does not contain com or dannce predictions
    """

    m1 = sio.loadmat(file_1)
    m2 = sio.loadmat(file_2)

    if "com" in m1.keys():
        error = np.mean(
            np.abs(m1["com"][:N_TEST_IMAGES, ...] - m2["com"][:N_TEST_IMAGES, ...])
        )
        return error < th
    elif "pred" in m2.keys():
        error = np.mean(
            np.abs(
                m1["pred"][:N_TEST_IMAGES, ...] - m2["pred"][:N_TEST_IMAGES, ...]
            )
        )
        return error < th
    else:
        raise Exception("Expected fields (pred, com) not found in inputs")

def train_setup():
    setup = "cp ./label3d_temp_dannce.mat ./alabel3d_temp_dannce.mat"
    os.system(setup)


class TestComTrain(absltest.TestCase):
    def test_com_train(self):
        train_setup()
        args = [
            "com-train",
            "config_com_mousetest.yaml",
            "--com-finetune-weights=../../demo/markerless_mouse_1/COM/weights/",
            "--downfac=8",
        ]
        with patch("sys.argv", args):
            cli.com_train_cli()

    def test_com_train_mono(self):
        train_setup()
        args = ["com-train", "config_com_mousetest.yaml", "--mono=True", "--downfac=8"]
        with patch("sys.argv", args):
            cli.com_train_cli()

class TestComPredict(absltest.TestCase):
    def test_com_predict(self):
        train_setup()
        args = ["com-predict", "config_com_mousetest.yaml"]
        with patch("sys.argv", args):
            cli.com_predict_cli()
        self.assertTrue(compare_predictions(
            "../touchstones/COM3D_undistorted_masternn.mat",
            "../../demo/markerless_mouse_1/COM/predict_test/com3d0.mat",
        ))

    def test_com_predict_3_cams(self):
        setup = "cp ./label3d_temp_dannce_3cam.mat ./alabel3d_temp_dannce.mat"
        os.system(setup)
        args = ["com-predict", "config_com_mousetest.yaml", "--downfac=4"]
        with patch("sys.argv", args):
            cli.com_predict_cli()

    def test_com_predict_5_cams(self):
        setup = "cp ./label3d_temp_dannce_5cam.mat ./alabel3d_temp_dannce.mat"
        os.system(setup)
        args = ["com-predict", "config_com_mousetest.yaml", "--downfac=2"]
        with patch("sys.argv", args):
            cli.com_predict_cli()

class TestDannceTrain(absltest.TestCase):
    def test_dannce_train_finetune_max(self):
        train_setup()
        args = [
            "dannce-train",
            "config_mousetest.yaml",
            "--net-type=MAX",
            "--dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/weights/weights.rat.MAX/",
        ]
        with patch("sys.argv", args):
            cli.dannce_train_cli()

    def test_dannce_train_finetune_avg(self):
        train_setup()
        args = [
            "dannce-train",
            "config_mousetest.yaml",
            "--net-type=AVG",
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

if __name__ == "__main__":
    log_file = 'log_file.txt'
    with open(log_file, "w") as f:
       runner = unittest.TextTestRunner(f)
       absltest.main(testRunner=runner)
