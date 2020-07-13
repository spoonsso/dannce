#!/bin/bash
set -e
python setup.py install
#Set of tests to run.
#
#List of tests:
#1) Train COMfinder on markerless mouse demo (two exp.yaml)
#	Because tf 1.x struggles with random seed reproducibility,
#	So it will be difficult to compare weight files directly.
#	This is potentially solved by removing data shuffleing in
#	a bespoke training script. But for now we will just make sure that
#	the COMfinder training runs without error over the mouse data
# 2) Predict with the the COMfinder using the 1.x weights and make sure
# 	predictiosn are identical
# 3) Train for a few epochs with train.DANNCE.py, trying both the finetune_MAX
# 	and finetune_AVG setups, the "new" network setup, the continued network setup, the
# 	continued weights only setup.
# 		Just make sure these run without error
# 4) predict 1000 samples with predict_DANNCE.py Test the AVG and MAX nets and make sure
# 	predictions are identical to dannce v0.1

cd tests/configs

echo "Testing COMfinder training"
com-train config_com_mousetest.yaml

echo "Testing COMfinder prediction"
com-predict config_com_mousetest.yaml
python ../compare_predictions.py ../touchstones/COM3D_undistorted_masternn.mat ../../demo/markerless_mouse_1/COM/predict_test/com3d.mat 0.001

echo "Testing DANNCE training, finetune_MAX"
dannce-train config_mousetest.yaml --net-type=MAX --dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/weights/weights.rat.MAX/

echo "Testing DANNCE training, finetune_AVG"
dannce-train config_mousetest.yaml --net-type=AVG --dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/weights/

echo "Testing DANNCE training, AVG net from scratch"
dannce-train config_mousetest.yaml --net=unet3d_big_expectedvalue --train-mode=new --n-channels-out=22

echo "Testing DANNCE training, MAX net from scratch"
dannce-train config_mousetest.yaml --net=unet3d_big --train-mode=new --n-channels-out=22 --dannce-finetune-weights="None"

echo "Testing DANNCE training, AVG net continued"
dannce-train config_mousetest.yaml --train-mode=continued --dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/train_results/AVG/

echo "Testing DANNCE training, MAX net continued"
dannce-train config_mousetest.yaml --net=finetune_MAX --train-mode=continued --dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/train_results/

cp ./label3d_temp_dannce.mat ./alabel3d_temp_dannce.mat
echo "Testing DANNCE AVG prediction"
dannce-predict config_mousetest.yaml
python ../compare_predictions.py ../touchstones/save_data_AVG_torch_nearest.mat ../../demo/markerless_mouse_1/DANNCE/predict_results/save_data_AVG.mat 0.001

echo "Testing DANNCE MAX prediction"
dannce-predict config_mousetest.yaml --expval=False --dannce-predict-model=../../demo/markerless_mouse_1/DANNCE/train_results/weights.12000-0.00014.hdf5
python ../compare_predictions.py ../touchstones/save_data_MAX_torchnearest_newtfroutine.mat ../../demo/markerless_mouse_1/DANNCE/predict_results/save_data_MAX.mat 0.001

# Remove temporary folders containign weights, etc.
# rm -rf ./DANNCE/
# rm -rf ./COM/
echo "PASSED WITHOUT ERROR"