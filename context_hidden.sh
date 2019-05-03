#!/usr/bin/env bash

python3 train.py --task 1 --model hidden --data personal --name task1_hidden --epochs 50 --cuda --log
python3 plot.py --file plot-data-task1_hidden.pkl
python3 train.py --task 2 --model hidden --data personal --name task2_hidden --epochs 50 --cuda --log
python3 plot.py --file plot-data-task2_hidden.pkl
python3 train.py --task 3 --model hidden --data personal --name task3_hidden --epochs 50 --cuda --log
python3 plot.py --file plot-data-task3_hidden.pkl
python3 train.py --task 4 --model hidden --data personal --name task4_hidden --epochs 50 --cuda --log
python3 plot.py --file plot-data-task4_hidden.pkl
python3 train.py --task 5 --model hidden --data personal --name task5_hidden --epochs 50 --cuda --log
python3 plot.py --file plot-data-task5_hidden.pkl
