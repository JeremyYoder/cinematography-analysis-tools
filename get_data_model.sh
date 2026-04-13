#!/bin/bash
set -euo pipefail

# Security enhancement: Fail securely on error, undefined variables, or pipeline failures
set -euo pipefail

echo "Creating directories to store the model & data"
mkdir -p models
mkdir -p train
mkdir -p valid

## ---- Download Models ----

echo "Downloading .pth model"
cd models/
gsutil cp gs://classifiermodel/*.pth .

## ---- Download Data ----

echo "Downloading training data"
cd ../train/
gsutil cp gs://classifiermodel/train.zip .
unzip train.zip
rm -f train.zip

echo "Downloading validation data"
cd ../valid/
gsutil cp gs://classifiermodel/valid.zip .
unzip valid.zip
rm -f valid.zip
