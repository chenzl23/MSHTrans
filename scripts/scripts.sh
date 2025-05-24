#!/bin/bash
python ./main.py --dataset-id SWaT --device 0  
python ./main.py --dataset-id WADI --device 0  
python ./main.py --dataset-id SMAP --device 0  
python ./main.py --dataset-id SMD --device 0   
python ./main.py --dataset-id MSL --device 0  --stride 1

