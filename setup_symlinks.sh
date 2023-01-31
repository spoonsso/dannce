#!/bin/bash

# 1st arg - DANNCE HOME location from where the weights and videos actually are downloaded
# 2nd arg - DANNCE HOME location to where the links are videos need to be linked to

cd  $2/demo/markerless_mouse_1;

rm -rf videos;
ln -s $1/demo/markerless_mouse_1/videos/ ./;

rm -rf DANNCE/train_results/*
ln -s $1/demo/markerless_mouse_1/DANNCE/train_results/* DANNCE/train_results/

rm -rf DANNCE/weights/*
ln -s $1/demo/markerless_mouse_1/DANNCE/weights/* DANNCE/weights/

cd  ../markerless_mouse_2;
rm -rf videos;
ln -s $1/demo/markerless_mouse_2/videos/ ./;