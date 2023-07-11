@echo off
cd C:\Users\BubbleTea\Desktop\Energy Forcasting\Time-Series-Library

python -u run.py ^
--tags StandardScaler,best ^
--neptune_id POW-78 ^
--no-train --model_id ID1 --run_id 003 --iter 1 ^
--data_source B

