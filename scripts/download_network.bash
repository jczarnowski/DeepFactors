#!/bin/bash
URL="https://www.dropbox.com/s/fj5dfq9hkqv2hm1/scannet256_32.tar.gz"
WEIGHT_PATH="data/nets"

if [[ ! -d ${WEIGHT_PATH} ]]; then
  # Download pretrained weights
  mkdir -p ${WEIGHT_PATH}
  wget -qO - ${URL} | tar xvz -C ${WEIGHT_PATH}
else
  echo "Pretrained weights have already exist in ${WEIGHT_PATH}!"
fi
