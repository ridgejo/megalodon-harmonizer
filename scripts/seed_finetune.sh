#! /bin/bash

# Three fixed random seeds selected randomly between 1 and 100
for i in 87 93 52;
do
    sbatch scripts/submit.sh train_rep.py --config "$1" --checkpoint "$2" --name "${3}_seed_${i}" --seed $i
done