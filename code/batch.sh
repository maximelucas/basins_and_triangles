#!/bin/bash

# Define values for your parameter
values=(15 20)

# Loop through the values and run the script with nohup
for val in "${values[@]}"; do
    nohup python -u basin_size_nb_randomHG.py --num_threads 4 -n 100 -t 600 -i RK45 -s "$val" > run_$val.out 2> run_$val.err &
    wait
done
