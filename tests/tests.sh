#!/bin/bash
set -e
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

python setup.py install

cd tests/configs

# echo "Testing COMfinder training"
# cp ./label3d_temp_dannce.mat ./alabel3d_temp_dannce.mat
# com-train config_com_mousetest.yaml --com-finetune-weights=../../demo/markerless_mouse_1/COM/weights/

# echo "Testing COMfinder training w/ mono"
# com-train config_com_mousetest.yaml --mono=True

# echo "Testing COMfinder prediction"
# com-predict config_com_mousetest.yaml
# python ../compare_predictions.py ../touchstones/COM3D_undistorted_masternn.mat ../../demo/markerless_mouse_1/COM/predict_test/com3d.mat 0.001

# echo "Testing COMfinder prediction, 3 cams"
# cp ./label3d_temp_dannce_3cam.mat ./alabel3d_temp_dannce.mat
# com-predict config_com_mousetest.yaml --downfac=4

# echo "Testing COMfinder prediction, 5 cams"
# cp ./label3d_temp_dannce_5cam.mat ./alabel3d_temp_dannce.mat
# com-predict config_com_mousetest.yaml --downfac=2

# echo "Testing DANNCE training, finetune_MAX"
cp ./label3d_temp_dannce.mat ./alabel3d_temp_dannce.mat
#dannce-train config_mousetest.yaml --net-type=MAX --dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/weights/weights.rat.MAX/

# echo "Testing DANNCE training, finetune_AVG"
# dannce-train config_mousetest.yaml --net-type=AVG --dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/weights/

# echo "Testing DANNCE training, AVG+MAX finetune fromMAX"
# dannce-train config_mousetest.yaml --avg-max=100 --dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/weights/weights.rat.MAX/

# echo "Testing DANNCE prediction after AVG+MAX finetune"
# dannce-predict config_mousetest.yaml --max-num-samples=100 --net-type=AVG --dannce-predict-model=../../demo/markerless_mouse_1/DANNCE/train_test/fullmodel_weights/fullmodel_end.hdf5


# echo "Testing DANNCE training, finetune_AVG with heatmap regularization"
# dannce-train config_mousetest.yaml --net-type=AVG --dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/weights/ --heatmap-reg=True

# echo "Testing DANNCE training, finetune_AVG from previous finetune"
# dannce-train config_mousetest.yaml --net-type=AVG --dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/train_results/AVG/

echo "Testing DANNCE training, AVG net from scratch"
dannce-train config_mousetest.yaml --net=unet3d_big_expectedvalue --train-mode=new --n-channels-out=22

# echo "Testing DANNCE training, MAX net from scratch"
# dannce-train config_mousetest.yaml --net=unet3d_big --train-mode=new --n-channels-out=22

# echo "Testing DANNCE training, AVG net continued"
# dannce-train config_mousetest.yaml --net-type=AVG --train-mode=continued --dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/train_results/AVG/

# echo "Testing DANNCE training, MAX net continued"
# dannce-train config_mousetest.yaml --net=finetune_MAX --train-mode=continued --dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/train_results/

# echo "Testing DANNCE training, AVG MONO from scratch"
# dannce-train config_mousetest.yaml --net-type=AVG --train-mode=new --net=unet3d_big_expectedvalue --mono=True --n-channels-out=22

# echo "Testing DANNCE training, AVG MONO from scratch w/ augmentation"
# dannce-train config_mousetest.yaml --net-type=AVG --train-mode=new --net=unet3d_big_expectedvalue --mono=True --n-channels-out=22 --augment-brightness=True --augment-continuous-rotation=True --augment-hue=True

# echo "Testing DANNCE training, AVG MONO finetune"
# dannce-train config_mousetest.yaml --net-type=AVG --mono=True --dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/weights/weights.rat.AVG.MONO/

# echo "Testing DANNCE training, AVG MONO finetune"
# dannce-train config_mousetest.yaml --net-type=AVG --mono=True --dannce-finetune-weights=../../demo/markerless_mouse_1/DANNCE/weights/weights.rat.AVG.MONO/ --drop-landmark=[5,7]

# cp ./label3d_temp_dannce.mat ./alabel3d_temp_dannce.mat
# echo "Testing DANNCE prediction, MONO"
# dannce-predict config_mousetest.yaml --net-type=AVG --dannce-predict-model=../../demo/markerless_mouse_1/DANNCE/train_test/fullmodel_weights/fullmodel_end.hdf5 --mono=True

# # 32 NVOX --------- 
# #MONO
# # echo "Testing DANNCE training, AVG MONO from scratch, 32 voxels, 3 cameras"
# # dannce-train config_mousetest.yaml --net-type=AVG --train-mode=new --net=unet3d_big_expectedvalue --mono=True --n-channels-out=22 --nvox=32 --n-views=3

# # cp ./label3d_temp_dannce_3cam.mat ./alabel3d_temp_dannce.mat

# # echo "Testing DANNCE prediction, MONO, 32 voxels, 3 cameras"
# # dannce-predict config_mousetest.yaml --net-type=AVG --dannce-predict-model=../../demo/markerless_mouse_1/DANNCE/train_test/fullmodel_weights/fullmodel_end.hdf5 --mono=True --nvox=32 --n-views=3

# #RGB
# # echo "Testing DANNCE training, AVG net from scratch, 32 voxels, 3 cameras"
# # dannce-train config_mousetest.yaml --net=unet3d_big_expectedvalue --train-mode=new --n-channels-out=22 --nvox=32 --n-views=3

# # cp ./label3d_temp_dannce_3cam.mat ./alabel3d_temp_dannce.mat

# # echo "Testing DANNCE AVG prediction"
# # dannce-predict config_mousetest.yaml --net-type=AVG --nvox=32 --n-views=3 --dannce-predict-model=../../demo/markerless_mouse_1/DANNCE/train_test/fullmodel_weights/fullmodel_end.hdf5

# # -----------

# cp ./label3d_temp_dannce.mat ./alabel3d_temp_dannce.mat
# echo "Testing DANNCE AVG prediction"
# dannce-predict config_mousetest.yaml --net-type=AVG
# python ../compare_predictions.py ../touchstones/save_data_AVG_torch_nearest.mat ../../demo/markerless_mouse_1/DANNCE/predict_test/save_data_AVG0.mat 0.001

# echo "Testing DANNCE MAX prediction"
# dannce-predict config_mousetest.yaml --net-type=MAX --expval=False --dannce-predict-model=../../demo/markerless_mouse_1/DANNCE/train_results/weights.12000-0.00014.hdf5
# python ../compare_predictions.py ../touchstones/save_data_MAX_torchnearest_newtfroutine.mat ../../demo/markerless_mouse_1/DANNCE/predict_test/save_data_MAX0.mat 0.001

# echo "Testing npy volume generation"
# cp ./label3d_voltest_dannce_m1.mat ./alabel3d_temp_dannce.mat
# dannce-predict config_mousetest.yaml --net-type=AVG --write-npy=../../demo/markerless_mouse_1/npy_volumes/ --batch-size=1

# cp ./label3d_voltest_dannce_m2.mat ./alabel3d_temp_dannce.mat
# dannce-predict config_mousetest.yaml --net-type=AVG --write-npy=../../demo/markerless_mouse_1/npy_volumes/ --batch-size=1

# echo "Testing npy volume training, with validation recording"
# cd ../../demo/markerless_mouse_1/
# dannce-train ../../configs/dannce_mouse_config.yaml --net-type=AVG --use-npy=True --dannce-train-dir=./DANNCE/npy_test/ --epochs=10 --valid-exp=[1] --gpu=1

# echo "Testing npy volume training, with multi gpu support"
# dannce-train ../../configs/dannce_mouse_config.yaml --net-type=AVG --batch-size=4 --use-npy=True --dannce-train-dir=./DANNCE/npy_test/ --epochs=10 --multi-gpu-train=True --gpu=1

# echo "Testing npy volume testing, with num_train_exp"
# dannce-train ../../configs/dannce_mouse_config.yaml --net-type=AVG --use-npy=True --dannce-train-dir=./DANNCE/npy_test/ --epochs=10 --num-train-per-exp=2 --batch-size=1 --gpu=1

# echo "Testing npy volume testing, with num_train_exp and validation recording"
# dannce-train ../../configs/dannce_mouse_config.yaml --net-type=AVG --use-npy=True --dannce-train-dir=./DANNCE/npy_test/ --epochs=10 --valid-exp=[1] --num-train-per-exp=2 --batch-size=1 --gpu=1

echo "PASSED WITHOUT ERROR"
