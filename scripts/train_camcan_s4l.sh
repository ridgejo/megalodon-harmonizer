#! /bin/bash

declare -a labels=("voice")
declare -a seeds=("1" "2" "3")
declare -a scale=("12")

for i in "${!labels[@]}"
do
    for j in "${!scale[@]}"
    do
        for k in "${!seeds[@]}"
        do
            sbatch scripts/submit_long_60h_256G.sh train_rep.py --config configs/neurips/pnpl/s4l/camcan_${labels[i]}/${scale[j]}/c${seeds[k]}.yaml --name neurips_camcan_s4l+${labels[i]}_${scale[j]}_seed${seeds[k]}
        done
    done
done