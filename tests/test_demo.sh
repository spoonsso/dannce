#!/bin/bash
set -e
#Tests each component of the demo
cd demo/markerless_mouse_1/
com-train ../../configs/com_mouse_config.yaml --epochs=3
dannce-train ../../configs/dannce_mouse_config.yaml --epochs=3
dannce-predict ../../configs/dannce_mouse_config.yaml

cp label3d_dannce.mat alabel3d_dannce.mat
com-predict ../../configs/com_mouse_config.yaml
rm alabel3d_dannce.mat

cd ../markerless_mouse_2/
dannce-predict ../../configs/dannce_mouse_config.yaml

cp label3d_dannce.mat alabel3d_dannce.mat
com-predict ../../configs/com_mouse_config.yaml
rm alabel3d_dannce.mat
