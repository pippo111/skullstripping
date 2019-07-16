#!/bin/sh
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/usr/src/app --runtime=nvidia -it --rm --user $(id -u):$(id -g) fbdev/ml:latest bash
