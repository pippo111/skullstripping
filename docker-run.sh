#!/bin/sh
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/usr/src/app --runtime=nvidia -it --rm --user $(id -u):$(id -g) fbdev/ml:latest bash
# python train.py --model-name=all_axis_0 --loss-function=dice_loss --image-width=256 --image-height=256
# tensorboard --logdir=logs/