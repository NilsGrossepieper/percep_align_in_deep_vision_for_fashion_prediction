#!/bin/bash

# Make sure the datasets folder exists
mkdir -p /content/drive/MyDrive/perceptual-vits-fashion-forecasting/datasets

# Change to datasets directory
cd /content/drive/MyDrive/perceptual-vits-fashion-forecasting/datasets

# Make sure the fashion_triplets folder exists
mkdir -p fashion_triplets

# Unzip the zip file into the fashion_triplets folder
unzip fashion_triplets.zip -d fashion_triplets

echo "Extraction complete!"
