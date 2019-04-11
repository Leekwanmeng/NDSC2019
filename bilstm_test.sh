#!/bin/bash

python3 bilstm.py --mode test --cat fashion_image --fold 1
python3 bilstm.py --mode test --cat mobile_image --fold 0
python3 bilstm.py --mode test --cat beauty_image --fold 3

