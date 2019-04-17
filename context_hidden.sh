#!/usr/bin/env bash

python3 train.py --model_personalized --data_personalized --log  --task personalized-dialog-task1-API-calls --name task1_hidden --epochs 1 --personalization_type 'hidden'
python3 plot.py --file plot-data-task1.pkl
python3 train.py --model_personalized --data_personalized --log  --task personalized-dialog-task2-API-refine --name task2_hidden --epochs 1 --personalization_type 'hidden'
python3 plot.py --file plot-data-task2.pkl
python3 train.py --model_personalized --data_personalized --log  --task personalized-dialog-task3-options --name task3_hidden --epochs 1 --personalization_type 'hidden'
python3 plot.py --file plot-data-task3.pkl
python3 train.py --model_personalized --data_personalized --log  --task personalized-dialog-task4-info --name task4_hidden --epochs 1 --personalization_type 'hidden'
python3 plot.py --file plot-data-task4.pkl
python3 train.py --model_personalized --data_personalized --log  --task personalized-dialog-task5-full-dialogs --name task5_hidden --epochs 1 --personalization_type 'hidden'
python3 plot.py --file plot-data-task5.pkl
