#!/bin/bash
set -e

# -------
# Run this bash script while inside demo/markerless_mouse_1/
# Note: this script will only work with --start-sample=0 (default) because the prediction
# output file is named `save_data_AVG{start-batch}`. To operate with a different start-sample,
# adjust the first argument to makeStructuredDataNoMocap.py accordingly.
# -------

# Run dannce predictions
dannce-predict ../../configs/dannce_mouse_config.yaml

# Generate predictions.mat from save_data_AVG0.mat
python ../../dannce/utils/makeStructuredDataNoMocap.py ./DANNCE/predict_results/save_data_AVG0.mat ../../configs/mouse22_skeleton.mat ./label3d_dannce.mat