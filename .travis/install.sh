#!/bin/sh

pip install -r requirements.txt
sudo add-apt-repository ppa:mc3man/trusty-media -y
sudo apt-get update
sudo apt-get install ffmpeg
