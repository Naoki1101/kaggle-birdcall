#!/bin/bash

mkdir -p data
mkdir -p data/input
mkdir -p data/input/freesound
mkdir -p data/input/birdcall_check
mkdir -p data/input/example_noise
mkdir -p logs
mkdir -p pickle

# download birdsong-recognition
cd ./data/input/
kaggle competitions download -c birdsong-recognition -p .
unzip ./*.zip

# download freesound-audio-tagging-2019
cd ./freesound
kaggle competitions download -c freesound-audio-tagging-2019 -f train_curated.csv -p .
kaggle competitions download -c freesound-audio-tagging-2019 -f train_curated.zip -p .
unzip train_curated.zip
rm -rf train_curated.zip

# # download birdcall-check
cd ../birdcall_check
kaggle datasets download -d shonenkov/birdcall-check -p .
unzip birdcall-check.zip
rm -rf birdcall-check.zip