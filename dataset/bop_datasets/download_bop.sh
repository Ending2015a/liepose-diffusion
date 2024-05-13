#!/bin/bash

SRC=https://bop.felk.cvut.cz/media/data/bop_datasets

LM=0
LMO=0
TLESS=1
YCBV=0
HB=0
ICBIN=0

if [ $LM -eq "1" ]; then
  wget $SRC/lm_base.zip
  wget $SRC/lm_models.zip
  wget $SRC/lm_train_pbr.zip
  wget $SRC/lm_test_all.zip
  wget $SRC/lm_train.zip

  unzip lm_base.zip
  unzip lm_models.zip -d lm
  unzip lm_train_pbr.zip -d lm
  unzip lm_test_all.zip -d lm
  unzip lm_train.zip -d lm
fi

if [ $LMO -eq "1" ]; then
  wget $SRC/lmo_base.zip
  wget $SRC/lmo_models.zip
  wget $SRC/lm_train_pbr.zip -O lmo_train_pbr.zip
  wget $SRC/lmo_test_all.zip
  wget $SRC/lmo_train.zip

  unzip lmo_base.zip
  unzip lmo_models.zip -d lmo
  unzip lmo_train_pbr.zip -d lmo
  unzip lmo_test_all.zip -d lmo
  unzip lmo_train.zip -d lmo
fi

if [ $TLESS -eq "1" ]; then
  wget $SRC/tless_base.zip
  wget $SRC/tless_models.zip
  wget $SRC/tless_train_pbr.zip
  wget $SRC/tless_train_primesense.zip
  wget $SRC/tless_test_primesense_all.zip

  unzip tless_base.zip
  unzip tless_models.zip -d tless
  unzip tless_train_pbr.zip -d tless
  unzip tless_train_primesense.zip -d tless
  unzip tless_test_primesense_all.zip -d tless
fi

if [ $YCBV -eq "1" ]; then
  wget $SRC/ycbv_base.zip
  wget $SRC/ycbv_models.zip
  wget $SRC/ycbv_train_pbr.zip
  wget $SRC/ycbv_train_real.zip
  wget $SRC/ycbv_test_all.zip

  unzip ycbv_base.zip
  unzip ycbv_models.zip -d ycbv
  unzip ycbv_train_pbr.zip -d ycbv
  unzip ycbv_train_real.zip -d ycbv
  unzip ycbv_test_all.zip -d ycbv
fi

if [ $HB -eq "1" ]; then
  wget $SRC/hb_base.zip
  wget $SRC/hb_models.zip
  wget $SRC/hb_train_pbr.zip
  wget $SRC/hb_val_primesense.zip
  wget $SRC/hb_test_primesense_all.zip

  unzip hb_base.zip
  unzip hb_models.zip -d hb
  unzip hb_train_pbr.zip -d hb
  unzip hb_val_primesense.zip -d hb
  unzip hb_test_primesense_all.zip -d hb
fi

if [ $ICBIN -eq "1" ]; then
  wget $SRC/icbin_base.zip
  wget $SRC/icbin_models.zip
  wget $SRC/icbin_train_pbr.zip
  wget $SRC/icbin_test_all.zip

  unzip icbin_base.zip
  unzip icbin_models.zip -d icbin
  unzip icbin_train_pbr.zip -d icbin
  unzip icbin_train.zip -d icbin
  unzip icbin_test_all.zip -d icbin
fi
