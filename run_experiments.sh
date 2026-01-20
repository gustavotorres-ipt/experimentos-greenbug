#!/usr/bin/env bash

MODELS="convnet resnet18 resnet101"

SPECS="melspec logmel l2m l3m"

for M in $MODELS; do
    for S in $SPECS; do
        echo "Running $M $S..."
        python treinar_rede.py -m "$M" -e "$S"
    done
done
