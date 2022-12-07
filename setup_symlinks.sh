#!/bin/bash

cd  $2/demo/markerless_mouse_1;

rm -rf videos;
ln -s $1/demo/markerless_mouse_1/videos/ ./;

rm -rf DANNCE/train_results/*
ln -s $1/demo/markerless_mouse_1/DANNCE/train_results/* DANNCE/train_results/

cd  ../markerless_mouse_2;
rm -rf videos;
ln -s $1/demo/markerless_mouse_2/videos/ ./;