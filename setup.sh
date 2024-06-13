#!/bin/sh
cat model/x* > model/pytorch_model.bin
rm model/x*
pip3 install -r requirements.txt

