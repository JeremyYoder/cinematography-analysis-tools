#!/bin/bash

echo "Creating directories to store the model & data"
mkdir models
mkdir train
mkdir valid

## ---- Download Models ----

echo "Downloading .pth model"
cd models/
gsutil cp gs://classifiermodel/*.pth .

## ---- Download Data ----

#echo "Downloading dummy training data"
#cd ../train/
#wget --max-redirect=20 -O train.zip https://www.dropbox.com/sh/c164dthxczdqwmd/AADJIKAYXp1wPkMFznQlZ1Cka?dl=0
#unzip train.zip
#rm train.zip

#echo "Downloading validation data"
#cd ../valid/
#wget --max-redirect=20 -O valid.zip https://www.dropbox.com/sh/d8rrrwg7zihzz7y/AAAnFE5jSEroVA0Q5u9ugecQa?dl=0
#unzip valid.zip
#rm valid.zip
