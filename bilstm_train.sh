#!/bin/bash

python3 bilstm.py --mode train --cat fashion_image
python3 bilstm.py --mode train --cat mobile_image
python3 bilstm.py --mode train --cat beauty_image
