#!/bin/bash
URL="https://drive.google.com/file/d/1CQetOB7yVUl8MZ2_hNBnJh3G4bPGGdr6"
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
