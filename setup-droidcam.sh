#!/bin/bash

# This script sets up DroidCam on a Linux system.

set -e  # Para o script imediatamente em caso de erro

sudo apt update
sudo apt install v4l2loopback-dkms linux-headers-$(uname -r)
sudo modprobe v4l2loopback devices=1 video_nr=2 card_label="DroidCam"
lsmod | grep v4l2loopback
ls -l /dev/video*
