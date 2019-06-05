#!/usr/bin/env bash

ROOT="$(pwd)/../.."

# activate venv
source $ROOT/venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$ROOT

# sudo mount /dev/sdb2 /mnt/HDD
python train_power_towers_2.py
