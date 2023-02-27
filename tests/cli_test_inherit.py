from absl.testing import absltest
import tensorflow as tf
import dannce.cli as cli
from cluster import grid, multi_gpu
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

DANNCE_HOME = os.path.dirname(os.getcwd())

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
    if os.getcwd() != os.path.join(DANNCE_HOME, "tests/configs"):
        os.chdir(os.path.join(DANNCE_HOME, "tests/configs"))
    setup = "cp ./label3d_temp_dannce.mat ./alabel3d_temp_dannce.mat"
    os.system(setup)

def train_setup_5cams():
    if os.getcwd() != os.path.join(DANNCE_HOME, "tests/configs"):
        os.chdir(os.path.join(DANNCE_HOME, "tests/configs"))
    setup = "mkdir ../../demo/markerless_mouse_1/videos5/;" + \
    "ln -s ../../../dannce/demo/markerless_mouse_1/videos/*[0-5] ../../demo/markerless_mouse_1/videos5;" + \
    "mkdir ../../demo/markerless_mouse_2/videos5/;" + \
    "ln -s ../../../dannce/demo/markerless_mouse_2/videos/*[0-5] ../../demo/markerless_mouse_2/videos5 "
    os.system(setup)


# class TestComTrain(absltest.TestCase):
#     def test_com_train(self):
#         train_setup()
#         args = [
#             "com-train",
#             "config_com_mousetest.yaml",
#             "--com-finetune-weights=../../demo/markerless_mouse_1/COM/weights/",
#             "--downfac=8",
#         ]
#         with patch("sys.argv", args):
#             cli.com_train_cli()

#     def test_com_train_mono(self):
#         train_setup()
#         args = ["com-train", "config_com_mousetest.yaml", "--mono=True", "--downfac=8"]
#         with patch("sys.argv", args):
#             cli.com_train_cli()

# class TestComPredict(absltest.TestCase):
#     def test_com_predict(self):
#         assert os.getcwd() == os.path.join(DANNCE_HOME, "tests/configs"), \
#             "Test not being performed from the intended configs folder {}".format(os.path.join(DANNCE_HOME, "tests/configs"))
#         train_setup()
#         args = ["com-predict", os.path.join(os.path.join(DANNCE_HOME, "tests/configs"),"config_com_mousetest.yaml")]
#         with patch("sys.argv", args):
#             cli.com_predict_cli()
#         self.assertTrue(compare_predictions(
#             "../touchstones/COM3D_undistorted_masternn.mat",
#             "../../demo/markerless_mouse_1/COM/predict_test/com3d0.mat",
#         ))

#     def test_com_predict_3_cams(self):
#         setup = "cp ./label3d_temp_dannce_3cam.mat ./alabel3d_temp_dannce.mat"
#         os.system(setup)
#         assert os.getcwd() == os.path.join(DANNCE_HOME, "tests/configs"), \
#             "Test not being performed from the intended configs folder {}".format(os.path.join(DANNCE_HOME, "tests/configs"))
#         args = ["com-predict", "config_com_mousetest.yaml", "--downfac=4"]
#         with patch("sys.argv", args):
#             cli.com_predict_cli()

#     def test_com_predict_5_cams(self):
#         setup = "cp ./label3d_temp_dannce_5cam.mat ./alabel3d_temp_dannce.mat"
#         os.system(setup)
#         assert os.getcwd() == os.path.join(DANNCE_HOME, "tests/configs"), \
#             "Test not being performed from the intended configs folder {}".format(os.path.join(DANNCE_HOME, "tests/configs"))
#         args = ["com-predict", "config_com_mousetest.yaml", "--downfac=2"]
#         with patch("sys.argv", args):
#             cli.com_predict_cli()

# class TestDannceTrain(absltest.TestCase):
#     com_args = [
#             "dannce-train",
#             "config_mousetest.yaml",
#         ]
    
#     def do_setup(self, caller_func):
#         if 'train' in caller_func:
#             train_setup()
#             return 0
        
#         return 1

#     def do_test(self):
#         from inspect import stack
#         import copy
#         caller_function = stack()[1].function
        
#         setup=self.do_setup(caller_function)
#         if setup!=0 :
#             print("Setup Incomplete")

#         args_=copy.deepcopy(TestDannceTrain.com_args)
#         args = self.get_args(args_, caller_function)
        
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()

#     def get_args(self, args, caller_function):
        
#         losses = ["mask_nan_keep_loss", "mask_nan_l1_loss", "gaussian_cross_entropy_loss"]
#         net_types = ["max", "avgmax", "avg"]
#         nets = ["unet3d_big_expectedvalue", "unet3d_big"]
#         train_modes = ["new","finetune","continued"]

#         for loss in losses:
#             if loss in caller_function:
#                 args.append("--loss={}".format(loss))

#         for net_type in net_types:
#             if net_type in caller_function and not "avgmax" in caller_function:
#                 args.append("--net-type={}".format(net_type.upper()))
#                 break
#             if "avgmax" in caller_function:
#                 args.append("--net-type={}".format("AVG"))
#                 args.append("--avg-max={}".format(10))

#         for train_mode in train_modes:
#             if train_mode in caller_function:
#                 args.append("--train_mode={}".format(train_mode))
                
#                 if train_mode!="new":
#                     avgmax_weights_dir = "../../demo/markerless_mouse_1/DANNCE/train_results/AVG/"
#                     avg_weights_dir = "../../demo/markerless_mouse_1/DANNCE/weights/"
#                     max_weights_dir = "../../demo/markerless_mouse_1/DANNCE/weights/weights.rat.MAX/"
#                     if caller_function.count("finetune") > 1:
#                         args.append("--dannce-finetune-weights={}".format(avgmax_weights_dir))
#                     elif train_mode != "continued":
#                         args.append("--dannce-finetune-weights={}".format(avg_weights_dir if net_type!="max" else max_weights_dir))
#                     else:
#                         args.append("--dannce-finetune-weights={}".format(avgmax_weights_dir if net_type!="max" else avg_weights_dir))
                
#                 else:
#                     args.append("--n-channels-out=22")
#                     args.append("--net={}".format("unet3d_big" if net_type == "max" else "unet3d_big_expectedvalue"))
        
#         if "mono" in caller_function:
#             args.append("--mono=True")

        
#         return args
    
#     def test_dannce_train_finetune_max(self):
#         train_setup()
#         assert os.getcwd() == os.path.join(DANNCE_HOME, "tests/configs"), \
#             "Test not being performed from the intended configs folder {}".format(os.path.join(DANNCE_HOME, "tests/configs"))
#         args = [
#             "dannce-train",
#             "config_mousetest.yaml",
#             "--net-type=MAX",
#             "--dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/weights/weights.rat.MAX/",
#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()

#     def test_dannce_train_finetune_avg(self):
#         train_setup()
#         assert os.getcwd() == os.path.join(DANNCE_HOME, "tests/configs"), \
#             "Test not being performed from the intended configs folder {}".format(os.path.join(DANNCE_HOME, "tests/configs"))
#         args = [
#             "dannce-train",
#             "config_mousetest.yaml",
#             "--net-type=AVG",
#             "--dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/weights/",
#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()
    
#     def test_dannce_train_finetune_avgmax(self):
#         train_setup()
#         assert os.getcwd() == os.path.join(DANNCE_HOME, "tests/configs"), \
#             "Test not being performed from the intended configs folder {}".format(os.path.join(DANNCE_HOME, "tests/configs"))
#         args = [
#             "dannce-train",
#             "config_mousetest.yaml",
#             "--net-type=AVG",
#             "--avg-max=10",
#             "--dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/weights/",
#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()
    

#     def test_dannce_train_finetune_avg_heatmap_regularization(self):
#         train_setup()
#         args = [
#             "dannce-train",
#             "config_mousetest.yaml",
#             "--net-type=AVG",
#             "--dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/weights/",
#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()

#     def test_dannce_train_finetune_avg_from_finetune(self):
#         train_setup()
#         args = [
#             "dannce-train",
#             "config_mousetest.yaml",
#             "--net-type=AVG",
#             "--dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/train_results/AVG/",
#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()
    
#     def test_dannce_train_finetune_avgmax_from_finetune(self):
#         train_setup()
#         assert os.getcwd() == os.path.join(DANNCE_HOME, "tests/configs"), \
#             "Test not being performed from the intended configs folder {}".format(os.path.join(DANNCE_HOME, "tests/configs"))
#         args = [
#             "dannce-train",
#             "config_mousetest.yaml",
#             "--net-type=AVG",
#             "--avg-max=10",
#             "--dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/train_results/AVG/",
#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()

#     def test_dannce_train_avg(self):
#         train_setup()
#         args = [
#             "dannce-train",
#             "config_mousetest.yaml",
#             "--net=unet3d_big_expectedvalue",
#             "--train-mode=new",
#             "--n-channels-out=22",
#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()

#     def test_dannce_train_max(self):
#         train_setup()
#         args = [
#             "dannce-train",
#             "config_mousetest.yaml",
#             "--net=unet3d_big",
#             "--train-mode=new",
#             "--n-channels-out=22",
#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()

#     def test_dannce_train_avg_continued(self):
#         train_setup()
#         args = [
#             "dannce-train",
#             "config_mousetest.yaml",
#             "--net-type=AVG",
#             "--train-mode=continued",
#             "--dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/train_results/AVG/",
#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()

#     def test_dannce_train_max_continued(self):
#         train_setup()
#         args = [
#             "dannce-train",
#             "config_mousetest.yaml",
#             "--net=finetune_MAX",
#             "--train-mode=continued",
#             "--dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/train_results/",
#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()

#     def test_dannce_train_avg_mono(self):
#         train_setup()
#         args = [
#             "dannce-train",
#             "config_mousetest.yaml",
#             "--net-type=AVG",
#             "--train-mode=new",
#             "--net=unet3d_big_expectedvalue",
#             "--mono=True",
#             "--n-channels-out=22",
#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()

#     def test_dannce_train_avg_mono_finetune(self):
#         train_setup()
#         args = [
#             "dannce-train",
#             "config_mousetest.yaml",
#             "--net-type=AVG",
#             "--mono=True",
#             "--dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/weights/weights.rat.AVG.MONO/",
#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()

#     def test_dannce_train_avg_mono_finetune_drop_landmarks(self):
#         train_setup()
#         args = [
#             "dannce-train",
#             "config_mousetest.yaml",
#             "--net-type=AVG",
#             "--mono=True",
#             "--dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/weights/weights.rat.AVG.MONO/",
#             "--drop-landmark=[5,7]",
#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()

#     def test_npy_volume_with_validation(self):
#         # train_setup()
#         os.chdir("../../demo/markerless_mouse_1/")
#         args = [
#             "dannce-train",
#             "../../configs/dannce_mouse_config.yaml",
#             "--net-type=AVG",
#             "--use-npy=True",
#             "--dannce-train-dir=./DANNCE/npy_test/",
#             "--epochs=2",
#             "--valid-exp=[1]",
#             "--gpu=1",
#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()

#     def test_npy_volume_with_multi_gpu(self):
#         # train_setup()
#         os.chdir("../../demo/markerless_mouse_1/")
#         args = [
#             "dannce-train",
#             "../../configs/dannce_mouse_config.yaml",
#             "--net-type=AVG",
#             "--batch-size=4",
#             "--use-npy=True",
#             "--dannce-train-dir=./DANNCE/npy_test/",
#             "--epochs=2",
#             "--valid-exp=[1]",
#             "--multi-gpu-train=True",
#             "--gpu=1",
#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()

#     def test_npy_volume_with_num_train_exp(self):
#         os.chdir("../../demo/markerless_mouse_1/")
#         args = [
#             "dannce-train",
#             "../../configs/dannce_mouse_config.yaml",
#             "--net-type=AVG",
#             "--use-npy=True",
#             "--dannce-train-dir=./DANNCE/npy_test/",
#             "--epochs=2",
#             "--num-train-per-exp=2",
#             "--batch-size=1",
#             "--gpu=1",
#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()

#     def test_npy_volume_with_validation_and_num_train_exp(self):
#         os.chdir("../../demo/markerless_mouse_1/")
#         args = [
#             "dannce-train",
#             "../../configs/dannce_mouse_config.yaml",
#             "--net-type=AVG",
#             "--use-npy=True",
#             "--dannce-train-dir=./DANNCE/npy_test/",
#             "--epochs=2",
#             "--valid-exp=[1]",
#             "--num-train-per-exp=2",
#             "--batch-size=1",
#             "--gpu=1",
#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()

#     def test_dannce_train_MAX_layer_norm(self):
#         train_setup()
#         args = [
#             "dannce-train",
#             "dgptest_config.yaml",
#             "--dannce-train-dir=../../demo/markerless_mouse_1/DANNCE/train_test_ln/",
#             "--n-channels-out=22",
#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()

#     def test_dannce_train_MAX_scratch_instance_norm(self):
#         train_setup()
#         args = [
#             "dannce-train",
#             "dgptest_config.yaml",
#             "--norm-method=instance",
#             "--dannce-train-dir=../../demo/markerless_mouse_1/DANNCE/train_test_in/",
#             "--n-channels-out=22",
#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()

#     def test_dannce_train_DGP_MAX_scratch_layer_norm_sigmoid_cross_entropy_Gaussian(
#         self,
#     ):
#         train_setup()
#         args = [
#             "dannce-train",
#             "dgptest_config.yaml",
#             "--loss=gaussian_cross_entropy_loss",
#             "--dannce-train-dir=../../demo/markerless_mouse_1/DANNCE/train_test_ln_dgp/",
#             "--n-channels-out=22",
#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()

#     def test_dannce_train_DGP_MAX_scratch_instance_norm_sigmoid_cross_entropy_Gaussian(
#         self,
#     ):
#         train_setup()
#         args = [
#             "dannce-train",
#             "dgptest_config.yaml",
#             "--norm-method=instance",
#             "--loss=gaussian_cross_entropy_loss",
#             "--dannce-train-dir=../../demo/markerless_mouse_1/DANNCE/train_test_ln_dgp/",
#             "--n-channels-out=22",
#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()

#     def test_dannce_train_avgmax(self):
#         train_setup()
#         args = [
#             "dannce-train",
#             "config_mousetest.yaml",
#             "--net-type=AVG",
#             "--avg-max=10",
#             "--dannce-train-dir=../../demo/markerless_mouse_1/DANNCE/train_test/AVG_MAX/", 
#             "--dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/train_results/AVG/",
#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()
    
#     def test_dannce_train_finetune_avg_grid(self):
#         train_setup()
#         args = [
#             "dannce-train-grid",
#             "config_mousetest.yaml",
#             "grid_config.yaml",
            
#         ]
#         with patch("sys.argv", args):
#             grid.dannce_train_grid()

#     def test_dannce_train_finetune_avg_6cams_with5camswts(self):
#         # This will probably fail when using save_pred_targets = True
#         # save_pred_targets based testing need to be done for both AVG and MAX models
#         train_setup()
#         args = [
#             "dannce-train",
#             "config_mousetest.yaml",
#             "--n-rand-views=5",
#             "--net-type=AVG",
#             "--mono=True",
#             "--dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/weights/weights.rat.AVG.MONO.5cams/",

#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()
    
#     def test_dannce_train_finetune_avg_5cams_with6camswts(self):
#         train_setup_5cams()
#         args = [
#             "dannce-train",
#             "config_mousetest.yaml",
#             "--io-config=io_5cams.yaml",
#             "--n-views=6",
#             "--net-type=AVG",
            
#         ]
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()

    
    
# class TestDannceTrainLosses(absltest.TestCase):
#     com_args = [
#             "dannce-train",
#             "config_mousetest.yaml",
#         ]
    
#     def do_setup(self, caller_func):
#         if 'train' in caller_func:
#             train_setup()
#             return 0
        
#         return 1

#     def do_test(self, spec_args=[]):
#         # import pdb; pdb.set_trace()
#         from inspect import stack
#         import copy
#         caller_function = stack()[1].function
        
#         setup=self.do_setup(caller_function)
#         if setup!=0 :
#             print("Setup Incomplete")
        
#         assert os.getcwd() == os.path.join(DANNCE_HOME, "tests/configs"), \
#             "Test not being performed from the intended configs folder {} and instead being performed from {}".format(os.path.join(DANNCE_HOME, "tests/configs"),
#                                                                                                                         os.getcwd())
        
#         args_=copy.deepcopy(TestDannceTrainLosses.com_args)
#         args = self.get_args(args_, caller_function)

#         if len(spec_args) > 0:
#             args.extend(spec_args)
        
#         with patch("sys.argv", args):
#             cli.dannce_train_cli()

#     def get_args(self, args, caller_function):
        
#         losses = ["mask_nan_keep_loss", "mask_nan_l1_loss", "gaussian_cross_entropy_loss" ]
#         distances = ["euclidean_distance_3D", "centered_euclidean_distance_3D"]
#         net_types = ["max", "avgmax", "avg"]
#         nets = ["unet3d_big_expectedvalue", "unet3d_big"]
#         train_modes = ["new","finetune","continued"]

        

#         for net_type in net_types:
#             if net_type in caller_function and not "avgmax" in caller_function:
#                 args.append("--net-type={}".format(net_type.upper()))
#                 break
#             if "avgmax" in caller_function:
#                 args.append("--net-type={}".format("AVG"))
#                 args.append("--avg-max={}".format(10))

#         for train_mode in train_modes:
#             if train_mode in caller_function:
#                 args.append("--train-mode={}".format(train_mode))
                
#                 if train_mode!="new":
#                     avgmax_weights_dir = "../../demo/markerless_mouse_1/DANNCE/train_results/AVG/"
#                     avg_weights_dir = "../../demo/markerless_mouse_1/DANNCE/weights/"
#                     max_weights_dir = "../../demo/markerless_mouse_1/DANNCE/weights/weights.rat.MAX/"
#                     if caller_function.count("finetune") > 1:
#                         args.append("--dannce-finetune-weights={}".format(avgmax_weights_dir))
#                     elif train_mode != "continued":
#                         args.append("--dannce-finetune-weights={}".format(avg_weights_dir if net_type!="max" else max_weights_dir))
#                     else:
#                         args.append("--dannce-finetune-weights={}".format(avgmax_weights_dir if net_type!="max" else avg_weights_dir))
                
#                 else:
#                     args.append("--n-channels-out=22")
#                     args.append("--net={}".format("unet3d_big" if net_type == "max" else "unet3d_big_expectedvalue"))
        
#         if "mono" in caller_function:
#             args.append("--mono=True")

#         for loss in losses:
#             if loss in caller_function:
#                 args.append("--loss={}".format(loss))
        
#         for distance in distances:
#             if distance in caller_function:
#                 args.append("--metric={}".format(distance))
        
#         return args

    
#     def test_dannce_train_finetune_avg_mask_nan_keep_loss(self):
#         self.do_test()
    
#     def test_dannce_train_finetune_avg_mask_nan_l1_loss(self):
#         self.do_test()
    
#     def test_dannce_train_finetune_avg_gaussian_cross_entropy_loss(self):
#         self.do_test()
    
#     def test_dannce_train_finetune_max_mask_nan_keep_loss(self):
#         self.do_test()
    
#     def test_dannce_train_finetune_max_mask_nan_l1_loss(self):
#         self.do_test()
    
#     def test_dannce_train_finetune_max_gaussian_cross_entropy_loss(self):
#         self.do_test()

        


    


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
    
    def test_dannce_predict_avgmax(self):
        train_setup()
        args = [
            "dannce-predict",
            "config_mousetest.yaml",
            "--net-type=AVG",
            "--dannce-predict-dir=../../demo/markerless_mouse_1/DANNCE/predict_test/AVG_MAX/",
            "--dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/train_results/AVG_MAX/"
        ]
        import pdb; pdb.set_trace()
        with patch("sys.argv", args):
            cli.dannce_predict_cli()
        self.assertTrue(compare_predictions(
            "../touchstones/save_data_AVG_MAX.mat",
            "../../demo/markerless_mouse_1/DANNCE/predict_test/AVG_MAX/save_data_AVG0.mat",
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
    log_file = 'log_file_inherit.txt'
    with open(log_file, "w") as f:
       runner = unittest.TextTestRunner(f)
       absltest.main(testRunner=runner)
