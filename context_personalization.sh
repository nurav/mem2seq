#!/usr/bin/env bash

#python3 train.py --model_personalized --data_personalized --log  --task personalized-dialog-task1-API-calls --name task1_context --epochs 50 --personalization_type 'context'
#python3 plot.py --file plot-data-task1_context.pkl
#echo 'plotting task 1 done!'
#python3 train.py --model_personalized --data_personalized --log  --task personalized-dialog-task2-API-refine --name task2_context --epochs 50 --personalization_type 'context'
#python3 plot.py --file plot-data-task2_context.pkl
#echo 'plotting task 2 done!'
python train.py --model_personalized --data_personalized --log  --task personalized-dialog-task3-options --name task3_context --epochs 50 --personalization_type 'context'
python plot.py --file plot-data-task3_context.pkl
echo 'plotting task 3 done!'
#python3 train.py --model_personalized --data_personalized --log  --task personalized-dialog-task4-info --name task4_context --epochs 50 --personalization_type 'context'
#python3 plot.py --file plot-data-task4_context.pkl
#echo 'plotting task 4 done!'
#python3 train.py --model_personalized --data_personalized --log  --task personalized-dialog-task5-full-dialogs --name task5_context --epochs 50 --personalization_type 'context'
#python3 plot.py --file plot-data-task5_context.pkl
#echo 'plotting task 5 done!'
