#!/bin/bash
URL="https://www.dropbox.com/s/fj5dfq9hkqv2hm1/scannet256_32.tar.gz"
DEST_DIR="data/nets"

if [[ ! -f ${DEST_DIR}/scannet256_32.cfg || 
      ! -f ${DEST_DIR}/scannet256_32_frozen.pb || 
      ! -f ${DEST_DIR}/scannet256_32_graphdef.pb ]]; then
  # Download pretrained weights
  mkdir -p ${DEST_DIR}
  wget -qO - ${URL} | tar xvz -C ${DEST_DIR}
else
  echo "The pretrained weights already exist"
fi
