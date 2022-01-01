"""Tests for locomotion.tasks.two_tap."""
import multi_gpu

import functools
from unittest.mock import patch
from absl.testing import absltest

import numpy as np
import os

DEMO_PATH = "../demo/markerless_mouse_1"
os.chdir(DEMO_PATH)
CONFIG_PATH = "../../tests/configs/config_mousetest.yaml"
MULTI_INSTANCE_CONFIG_PATH = "../../tests/configs/config_mousetest_multi_instance.yaml"
DANNCE_PATH = "../../tests/configs/label3d_dannce.mat"


class MultiGpuTest(absltest.TestCase):
    def test_dannce_predict_help_message(self):
        os.system("dannce-predict-multi-gpu --help")

    def test_com_predict_help_message(self):
        os.system("com-predict-multi-gpu --help")

    def test_dannce_predict_batch_params(self):
        handler = multi_gpu.MultiGpuHandler(
            CONFIG_PATH,
            n_samples_per_gpu=100,
            verbose=False,
            test=True,
            dannce_file=DANNCE_PATH,
        )
        batch_params, _ = handler.submit_dannce_predict_multi_gpu()
        self.assertTrue(os.path.exists(handler.batch_param_file))
        self.assertTrue(len(batch_params) == 10)

    def test_dannce_predict_batch_params_multi_instance(self):
        handler = multi_gpu.MultiGpuHandler(
            MULTI_INSTANCE_CONFIG_PATH,
            n_samples_per_gpu=100,
            verbose=False,
            test=True,
            dannce_file=DANNCE_PATH,
        )
        batch_params, _ = handler.submit_dannce_predict_multi_gpu()
        print(batch_params)
        self.assertTrue(len(batch_params) == 20)

    def test_dannce_inference_submission(self):
        with patch("sys.argv", ["dannce-inference", MULTI_INSTANCE_CONFIG_PATH, MULTI_INSTANCE_CONFIG_PATH, "--test=True"]):
            multi_gpu.submit_inference()

    def test_com_predict_batch_params(self):
        handler = multi_gpu.MultiGpuHandler(
            CONFIG_PATH,
            n_samples_per_gpu=100,
            verbose=False,
            test=True,
            dannce_file=DANNCE_PATH,
        )
        batch_params, _ = handler.submit_com_predict_multi_gpu()
        self.assertTrue(os.path.exists(handler.batch_param_file))
        self.assertTrue(len(batch_params) == 180)

    def test_raises_error_if_no_dannce_file(self):
        # Move to a directory in which there is no dannce.mat file
        os.chdir("..")
        with self.assertRaises(FileNotFoundError):
            handler = multi_gpu.MultiGpuHandler(
                CONFIG_PATH, n_samples_per_gpu=100, verbose=False, test=True
            )

    def test_dannce_predict_multi_gpu_cli(self):
        cmd = (
            "dannce-predict-multi-gpu %s --test=True --verbose=False --dannce-file=%s"
            % (
                CONFIG_PATH,
                DANNCE_PATH,
            )
        )
        os.system(cmd)

    def test_com_predict_multi_gpu_cli(self):
        cmd = (
            "com-predict-multi-gpu %s --test=True --verbose=False --dannce-file=%s"
            % (
                CONFIG_PATH,
                DANNCE_PATH,
            )
        )
        os.system(cmd)


if __name__ == "__main__":
    absltest.main()
