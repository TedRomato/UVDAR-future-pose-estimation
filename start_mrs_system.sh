#!/bin/bash

xhost +local:root

docker run -it \
  --rm \
  --user root \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$(pwd):/root" \
  ctumrs/mrs_uav_system:latest \
  bash --rcfile /root/.bashrc

