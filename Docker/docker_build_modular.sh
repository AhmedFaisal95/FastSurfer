#!/bin/bash
cd ..

# Default flag values
cuda="1"
python_version="3.6"
seg_only="0"
surf_only="0"

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    --no_cuda)
      cuda="0"
      shift # past argument
      ;;
    --seg_only)
      seg_only="1"
      shift # past argument
      ;;
    --surf_only)
      surf_only="1"
      shift # past argument
      ;;
    -p|--python_version)
      python_version="$2"
      shift # past argument
      shift # past value
      ;;
      *)    # unknown option
      echo ERROR: Flag $key unrecognized.
      exit 1
    ;;
  esac
done
## Source: https://stackoverflow.com/a/14203146

base_image=""
if [ "$cuda" == "1" ]
  then
      base_image="nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04"
  else
      base_image="ubuntu:16.04"
fi
echo "Building from base image: $base_image"

docker build --rm=true --build-arg BASE_IMAGE=$base_image \
    --build-arg CUDA=$cuda \
    # --build-arg PYTHON_VERSION=$python_version \
    --build-arg SEG_ONLY=$seg_only \
    --build-arg SURF_ONLY=$surf_only \
    --no-cache \
    -t fastsurfer:docker_mod_tests_fscnn -f ./Docker/Dockerfile_all .

# ---------------------------------------------------------------------




# if [ "$surf_only" == "0" ]
#   then
#       docker build --rm=true --build-arg BASE_IMAGE=$base_image \
#           --build-arg CUDA=$cuda \
#           # --build-arg PYTHON_VERSION=$python_version \
#           --build-arg SEG_ONLY=$seg_only \
#           --build-arg SURF_ONLY=$surf_only \
#           --no-cache \
#           -t fastsurfer:docker_mod_tests_fscnn -f ./Docker/Dockerfile_general .
#   else
#       docker build --rm=true --build-arg BASE_IMAGE=$base_image \
#           --build-arg CUDA=$cuda \
#           # --build-arg PYTHON_VERSION=$python_version \
#           --no-cache \
#           -t fastsurfer:docker_mod_tests_fscnn -f ./Docker/Dockerfile_general_surf .
# fi
