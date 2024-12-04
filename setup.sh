#!/bin/sh

# Skapa model/pytorch_model.bin berre éin gong, elles vert det overskrive.
ls model/x* &>/dev/null
if [[ "$?" -eq 0 ]]; then
    cat model/x* > model/pytorch_model.bin
    rm model/x*
fi

pip3 install -r requirements.txt
