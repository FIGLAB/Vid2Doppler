#!/usr/bin/env bash

echo "Creating virtual environment"
python3.7 -m venv vis_velocity
echo "Activating virtual environment"

source $PWD/vis_velocity/bin/activate

$PWD/vis_velocity/bin/pip install numpy==1.17.5 torch==1.4.0 torchvision==0.5.0
$PWD/vis_velocity/bin/pip install git+https://github.com/giacaglia/pytube.git --upgrade
$PWD/vis_velocity/bin/pip install -r requirements.txt
