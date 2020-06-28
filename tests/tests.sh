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

cd tests/configs

# echo "Testing COMfinder training"
# com-train config_com_mousetest.yaml

# echo "Testing COMfinder prediction"
# com-predict config_com_mousetest.yaml
# python ../compare_predictions.py ../touchstones/COM3D_undistorted_masternn.mat ../../demo/markerless_mouse_1/COM/predict_test/COM3D_undistorted.mat 0.001

echo "Testing DANNCE training, finetune_MAX"
awk '/net/{gsub(/finetune_AVG/, "finetune_MAX")};{print}' config_mousetest.yaml > config_temp.yaml
awk '/EXPVAL/{gsub(/True/, "False")};{print}' config_temp.yaml > config_temp2.yaml
awk '/weights/{gsub("../../demo/markerless_mouse_1/DANNCE/weights/", "../../demo/markerless_mouse_1/DANNCE/weights/weights.rat.MAX/")};{print}' config_temp2.yaml > base_config_temp.yaml
rm config_temp2.yaml
rm config_temp.yaml
dannce-train base_config_temp.yaml

echo "Testing DANNCE training, finetune_AVG"
awk '/net/{gsub(/finetune_AVG/, "finetune_AVG")};{print}' config_mousetest.yaml > base_config_temp.yaml
dannce-train base_config_temp.yaml

echo "Testing DANNCE training, AVG net from scratch"
awk '/net/{gsub(/finetune_AVG/, "unet3d_big_expectedvalue")};{print}' config_mousetest.yaml > config_temp.yaml
awk '/train_mode/{gsub(/finetune/, "new")};{print}' config_temp.yaml > config_temp2.yaml
awk '/N_CHANNELS_OUT/{gsub(/20/, "22")};{print}' config_temp2.yaml > base_config_temp.yaml
rm config_temp2.yaml
rm config_temp.yaml
dannce-train base_config_temp.yaml

echo "Testing DANNCE training, MAX net from scratch"
awk '/net/{gsub(/finetune_AVG/, "unet3d_big")};{print}' config_mousetest.yaml > config_temp.yaml
awk '/EXPVAL/{gsub(/True/, "False")};{print}' config_temp.yaml > config_temp2.yaml
awk '/train_mode/{gsub(/finetune/, "new")};{print}' config_temp2.yaml > config_temp3.yaml
awk '/N_CHANNELS_OUT/{gsub(/20/, "22")};{print}' config_temp3.yaml > config_temp2.yaml
awk '/weights/{gsub("../../demo/markerless_mouse_1/DANNCE/weights/", "None")};{print}' config_temp2.yaml > base_config_temp.yaml
rm config_temp.yaml
rm config_temp2.yaml
rm config_temp3.yaml
dannce-train base_config_temp.yaml

echo "Testing DANNCE training, AVG net continued"
awk '/train_mode/{gsub(/finetune/, "continued")};{print}' config_mousetest.yaml > config_temp2.yaml
awk '/weights/{gsub("../../demo/markerless_mouse_1/DANNCE/weights/", "../../demo/markerless_mouse_1/DANNCE/train_results/AVG/")};{print}' config_temp2.yaml > base_config_temp.yaml
rm config_temp2.yaml
dannce-train base_config_temp.yaml

echo "Testing DANNCE training, MAX net continued"
awk '/train_mode/{gsub(/finetune/, "continued")};{print}' config_mousetest.yaml > config_temp.yaml
awk '/EXPVAL/{gsub(/True/, "False")};{print}' config_temp.yaml > config_temp2.yaml
awk '/weights/{gsub("../../demo/markerless_mouse_1/DANNCE/weights/", "../../demo/markerless_mouse_1/DANNCE/train_results/")};{print}' config_temp2.yaml > base_config_temp.yaml
rm config_temp2.yaml
dannce-train config_mousetest.yaml

echo "Testing DANNCE MAX prediction"
awk '/EXPVAL/{gsub(/True/, "False")};{print}' config_mousetest.yaml > config_temp2.yaml
awk '/#predict_model/{gsub("#predict_model: path_to_model_file", "predict_model: ../../demo/markerless_mouse_1/DANNCE/train_results/weights.12000-0.00014.hdf5")};{print}' config_temp2.yaml > base_config_temp.yaml
rm config_temp2.yaml
dannce-predict base_config_temp.yaml
python ../compare_predictions.py ../touchstones/save_data_MAX_torchnearest_newtfroutine.mat ../../demo/markerless_mouse_1/DANNCE/predict_results/save_data_MAX.mat 0.001

echo "Testing DANNCE AVG prediction"
awk '/#predict_model/{gsub("#predict_model: path_to_model_file", "predict_model: ../../demo/markerless_mouse_1/DANNCE/train_results/AVG/weights.1200-12.77642.hdf5")};{print}' config_mousetest.yaml > base_config_temp.yaml
dannce-predict base_config_temp.yaml
python ../compare_predictions.py ../touchstones/save_data_AVG_torch_nearest.mat ../../demo/markerless_mouse_1/DANNCE/predict_results/save_data_AVG.mat 0.001

# Remove temporary folders containign weights, etc.
# rm -rf ./DANNCE/
# rm -rf ./COM/
echo "PASSED WITHOUT ERROR"