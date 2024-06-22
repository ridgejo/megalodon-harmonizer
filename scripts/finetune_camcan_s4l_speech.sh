#! /bin/bash

declare -a labels=("speech")
declare -a task=("voicing")
# declare -a seeds=("1" "2" "3" "1" "2" "3" "1" "2" "3")
# declare -a scale=("1" "1" "1" "3" "3" "3" "6" "6" "6" "12" "12" "12")
# declare -a ckpts=("w5jmmye9" "a1edi6ai" "ui7bewak" "38hg170l" "uzxbe5ig" "wyi1mtgi" "qnv3xamh" "urzy1hp6" "cy2mi9yz" "1xf7ron3" "dled1tat" "ni5ii72x")

declare -a seeds=("1" "2" "3")
declare -a scale=("27" "27" "27")
declare -a ckpts=("w4klyvka" "obcgf3ij" "lrj228it")

for i in "${!labels[@]}"
do
    for j in "${!scale[@]}"
    do
        sbatch scripts/submit_short_128G.sh train_rep.py --config configs/neurips/pnpl/s4l/camcan_${labels[i]}/${task[i]}.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ckpts[j]} --name neurips_camcan_s4l+${labels[i]}_${scale[j]}_task_${task[i]}_seed${seeds[j]} --seed ${seeds[j]}
    done
done