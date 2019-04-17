#!/usr/bin/env bash

python train.py --model_personalized --data_personalized --log  --task personalized-dialog-task1-API-calls --name task1_context --epochs 1 --personalization_type 'context'
python plot.py --file plot-data-task1.pkl
python train.py --model_personalized --data_personalized --log  --task personalized-dialog-task2-API-refine --name task2_context --epochs 1 --personalization_type 'context'
python plot.py --file plot-data-task2.pkl
python train.py --model_personalized --data_personalized --log  --task personalized-dialog-task3-options --name task3_context --epochs 1 --personalization_type 'context'
python plot.py --file plot-data-task3.pkl
python train.py --model_personalized --data_personalized --log  --task personalized-dialog-task4-info --name task4_context --epochs 1 --personalization_type 'context'
python plot.py --file plot-data-task4.pkl
python train.py --model_personalized --data_personalized --log  --task personalized-dialog-task5-full-dialogs --name task5_context --epochs 1 --personalization_type 'context'
python plot.py --file plot-data-task5.pkl
