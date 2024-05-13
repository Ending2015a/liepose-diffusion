#!/bin/bash

set -x

wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar --no-check-certificate
wget https://pjreddie.com/media/files/VOC2012test.tar --no-check-certificate
wget https://pjreddie.com/media/files/VOCdevkit_18-May-2011.tar --no-check-certificate

tar xvf VOCtrainval_11-May-2012.tar
tar xvf VOC2012test.tar
tar xvf VOCdevkit_18-May-2011.tar

set +x
