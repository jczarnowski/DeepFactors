#!/bin/bash
mkdir -p data/nets
URL="https://www.dropbox.com/s/fj5dfq9hkqv2hm1/scannet256_32.tar.gz"
wget -qO - ${URL} | tar xvz -C data/nets 
