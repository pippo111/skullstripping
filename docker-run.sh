#!/bin/sh
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/usr/src/app --runtime=nvidia -it --rm --user $(id -u):$(id -g) fbdev/ml:latest bash
# python train.py --model-name=all_axis_0 --loss-function=dice_loss
# python evaluate.py --model-name=all_axis_0 --loss-function=dice_loss --save-slice=True --slice-numbers=20,44,60,75,89,90,245
# tensorboard --logdir=logs/