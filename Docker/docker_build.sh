#!/bin/bash
cd ..
docker build --rm=true -t fastsurfer:gpu -f ./Docker/fastsurfer/gpu/Dockerfile .
