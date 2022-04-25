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
#gsutil cp gs://classifiermodel/train.zip .
#unzip train.zip
#rm train.zip

#echo "Downloading validation data"
#cd ../valid/
#gsutil cp gs://classifiermodel/valid.zip .
#unzip valid.zip
#rm valid.zip
