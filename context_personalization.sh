#!/usr/bin/env bash

python3 main.py --task 1 --model personal_context --data personal_context --name task1_context --epochs 61 --cuda --log
python3 plot.py --file plot-data-task1_context.pkl
python3 train.py --task 2 --model personal_context --data personal_context --name task2_context --epochs 61 --cuda --log
python3 plot.py --file plot-data-task2_context.pkl
python3 train.py --task 3 --model personal_context --data personal_context --name task3_context --epochs 61 --cuda --log
python3 plot.py --file plot-data-task3_context.pkl
python3 train.py --task 4 --model personal_context --data personal_context --name task4_context --epochs 61 --cuda --log
python3 plot.py --file plot-data-task4_context.pkl
python3 train.py --task 5 --model personal_context --data personal_context --name task5_context --epochs 61 --cuda --log
python3 plot.py --file plot-data-task5_context.pkl
