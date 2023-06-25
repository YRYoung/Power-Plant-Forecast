@echo off
cd C:\Users\BubbleTea\Desktop\Energy Forcasting\Time-Series-Library

python -u run.py ^
--tags StandardScaler,sequential_datasets ^
--train --model_id ID1 --run_id 005_C ^
--data_source C

python -u run.py ^
--tags StandardScaler,sequential_datasets ^
--train --model_id ID1 --run_id 005_all ^
--data_source all