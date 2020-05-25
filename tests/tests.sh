#!/bin/bash
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
cd tests
# python ../train_COMfinder.py config_mousetest.yaml

echo "Testing COMfinder prediction"
python ../predict_COMfinder.py config_mousetest.yaml
python compare_predictions.py ./touchstones/COM3D_undistorted_masternn.mat ./COM/predict_results/COM3D_undistorted.mat

# echo "Testing DANNCE training, finetune_MAX"
