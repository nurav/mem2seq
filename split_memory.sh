#!/usr/bin/env bash

python3 train.py --task 1 --model split_mem --data personal --name task1_split --epochs 100 --cuda --log
python3 plot.py --file plot-data-task1_split.pkl
python3 train.py --task 2 --model split_mem --data personal --name task2_split --epochs 100 --cuda --log
python3 plot.py --file plot-data-task2_split.pkl
python3 train.py --task 3 --model split_mem --data personal --name task3_split --epochs 100 --cuda --log
python3 plot.py --file plot-data-task3_split.pkl
python3 train.py --task 4 --model split_mem --data personal --name task4_split --epochs 100 --cuda --log
python3 plot.py --file plot-data-task4_split.pkl
python3 train.py --task 5 --model split_mem --data personal --name task5_split --epochs 100 --cuda --log
python3 plot.py --file plot-data-task5_split.pkl
