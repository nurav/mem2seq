#!/usr/bin/env bash

python3 train.py --model_personalized --data_personalized --log  --task personalized-dialog-task1-API-calls --name task1_split --epochs 100
python3 plot.py --file plot-data-task1_split.pkl
python3 train.py --model_personalized --data_personalized --log  --task personalized-dialog-task2-API-refine --name task2_split --epochs 100
python3 plot.py --file plot-data-task2.pkl
python3 train.py --model_personalized --data_personalized --log  --task personalized-dialog-task3-options --name task3_split --epochs 100
python3 plot.py --file plot-data-task3.pkl
python3 train.py --model_personalized --data_personalized --log  --task personalized-dialog-task4-info --name task4_split --epochs 100
python3 plot.py --file plot-data-task4.pkl
python3 train.py --model_personalized --data_personalized --log  --task personalized-dialog-task5-full-dialogs --name task5_split --epochs 100
python3 plot.py --file plot-data-task5.pkl
