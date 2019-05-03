#!/usr/bin/env bash

python3 train.py --task 1 --model personal_context --data personal --name task1_context --epochs 50 --cuda --log
python3 plot.py --file plot-data-task1_split.pkl
python3 train.py --task 2 --model personal_context --data personal --name task2_context --epochs 50 --cuda --log
python3 plot.py --file plot-data-task2.pkl
python3 train.py --task 3 --model personal_context --data personal --name task3_context --epochs 50 --cuda --log
python3 plot.py --file plot-data-task3.pkl
python3 train.py --task 4 --model personal_context --data personal --name task4_context --epochs 50 --cuda --log
python3 plot.py --file plot-data-task4.pkl
python3 train.py --task 5 --model personal_context --data personal --name task5_context --epochs 50 --cuda --log
python3 plot.py --file plot-data-task5.pkl
