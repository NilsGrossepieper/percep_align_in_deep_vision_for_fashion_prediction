#!/bin/bash

# Make sure the datasets folder exists
mkdir -p /content/drive/MyDrive/perceptual-vits-fashion-forecasting/datasets

# Change to datasets directory
cd /content/drive/MyDrive/perceptual-vits-fashion-forecasting/datasets

# Download the NIGHTS dataset
wget -O NIGHTS.zip https://data.csail.mit.edu/nights/nights.zip

# Unzip the dataset
unzip NIGHTS.zip

echo "Download and extraction complete!"
