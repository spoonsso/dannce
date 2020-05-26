#!/bin/bash
set -e
#Set of tests to run with travis CI. Called when running built tests.
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
# 	predictiosn are identical to dannce v0.1


# echo "Testing COMfinder training"
cd tests/configs
# python ../train_COMfinder.py config_mousetest.yaml

# echo "Testing COMfinder prediction"
# python ../predict_COMfinder.py config_mousetest.yaml
# python compare_predictions.py ./touchstones/COM3D_undistorted_masternn.mat ./COM/predict_results/COM3D_undistorted.mat 0.1

# echo "Testing DANNCE training, finetune_MAX"
# awk '/net/{gsub(/finetune_AVG/, "finetune_MAX")};{print}' config_DANNCEtest.yaml > config_temp.yaml
# awk '/EXPVAL/{gsub(/True/, "False")};{print}' config_temp.yaml > config_temp2.yaml
# awk '/weights/{gsub("../../demo/markerless_mouse_1/DANNCE/weights/", "../../demo/markerless_mouse_1/DANNCE/weights/weights.rat.MAX/")};{print}' config_temp2.yaml > config_temp.yaml
# rm config_temp2.yaml
# python ../../train_DANNCE.py config_mousetest.yaml

# echo "Testing DANNCE training, finetune_AVG"
# awk '/net/{gsub(/finetune_AVG/, "finetune_AVG")};{print}' config_DANNCEtest.yaml > config_temp.yaml
# python ../../train_DANNCE.py config_mousetest.yaml

# echo "Testing DANNCE training, AVG net from scratch"
# awk '/net/{gsub(/finetune_AVG/, "unet3d_big_expectedvalue")};{print}' config_DANNCEtest.yaml > config_temp.yaml
# awk '/train_mode/{gsub(/finetune/, "new")};{print}' config_temp.yaml > config_temp2.yaml
# awk '/N_CHANNELS_OUT/{gsub(/20/, "22")};{print}' config_temp2.yaml > config_temp.yaml
# rm config_temp2.yaml
# python ../../train_DANNCE.py config_mousetest.yaml

# echo "Testing DANNCE training, MAX net from scratch"
# awk '/net/{gsub(/finetune_AVG/, "unet3d_big")};{print}' config_DANNCEtest.yaml > config_temp.yaml
# awk '/EXPVAL/{gsub(/True/, "False")};{print}' config_temp.yaml > config_temp2.yaml
# awk '/train_mode/{gsub(/finetune/, "new")};{print}' config_temp2.yaml > config_temp3.yaml
# awk '/N_CHANNELS_OUT/{gsub(/20/, "22")};{print}' config_temp3.yaml > config_temp2.yaml
# awk '/weights/{gsub("../../demo/markerless_mouse_1/DANNCE/weights/", "None")};{print}' config_temp2.yaml > config_temp.yaml
# rm config_temp2.yaml
# rm config_temp3.yaml
# python ../../train_DANNCE.py config_mousetest.yaml

echo "Testing DANNCE training, AVG net continued"
awk '/train_mode/{gsub(/finetune/, "continued")};{print}' config_DANNCEtest.yaml > config_temp2.yaml
awk '/weights/{gsub("../../demo/markerless_mouse_1/DANNCE/weights/", "../../demo/markerless_mouse_1/DANNCE/train_results/AVG/")};{print}' config_temp2.yaml > config_temp.yaml
rm config_temp2.yaml
python ../../train_DANNCE.py config_mousetest.yaml

echo "Testing DANNCE training, MAX net continued"
awk '/train_mode/{gsub(/finetune/, "continued")};{print}' config_DANNCEtest.yaml > config_temp.yaml
awk '/EXPVAL/{gsub(/True/, "False")};{print}' config_temp.yaml > config_temp2.yaml
awk '/weights/{gsub("../../demo/markerless_mouse_1/DANNCE/weights/", "../../demo/markerless_mouse_1/DANNCE/train_results/")};{print}' config_temp2.yaml > config_temp.yaml
rm config_temp2.yaml
python ../../train_DANNCE.py config_mousetest.yaml

echo "Testing DANNCE training, MAX net, continued weights only"
awk '/train_mode/{gsub(/finetune/, "continued_weights_only")};{print}' config_DANNCEtest.yaml > config_temp.yaml
awk '/EXPVAL/{gsub(/True/, "False")};{print}' config_temp.yaml > config_temp2.yaml
awk '/weights/{gsub("../../demo/markerless_mouse_1/DANNCE/weights/", "../../demo/markerless_mouse_1/DANNCE/train_results/")};{print}' config_temp2.yaml > config_temp.yaml
awk '/net/{gsub(/finetune_AVG/, "unet3d_big")};{print}' config_temp.yaml > config_temp3.yaml
awk '/N_CHANNELS_OUT/{gsub(/20/, "22")};{print}' config_temp3.yaml > config_temp.yaml
rm config_temp2.yaml config_temp3.yaml
python ../../train_DANNCE.py config_mousetest.yaml

echo "Testing DANNCE training, AVG net, continued weights only"
awk '/train_mode/{gsub(/finetune/, "continued_weights_only")};{print}' config_DANNCEtest.yaml > config_temp.yaml
awk '/weights/{gsub("../../demo/markerless_mouse_1/DANNCE/weights/", "../../demo/markerless_mouse_1/DANNCE/train_results/AVG/")};{print}' config_temp.yaml > config_temp2.yaml
awk '/net/{gsub(/finetune_AVG/, "unet3d_big_expectedvalue")};{print}' config_temp2.yaml > config_temp3.yaml
awk '/N_CHANNELS_OUT/{gsub(/20/, "22")};{print}' config_temp3.yaml > config_temp.yaml
rm config_temp2.yaml config_temp3.yaml
python ../../train_DANNCE.py config_mousetest.yaml

echo "PASSED WITHOUT ERROR"