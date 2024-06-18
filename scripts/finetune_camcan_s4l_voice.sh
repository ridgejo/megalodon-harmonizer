#! /bin/bash

declare -a labels=("voice")
declare -a task=("speech")
# declare -a seeds=("1" "2" "3" "1" "2" "3" "1" "2" "3")
# declare -a scale=("1" "1" "1" "3" "3" "3" "6" "6" "6")
# declare -a ckpts=("porhzfh9" "3dpyyazu" "brnxd0h8" "a0b4s4q0" "99p3amf8" "me83qk5j" "2e3jw2bb" "lmu1hyed" "0glbj1wb")

# declare -a seeds=("1" "2" "3")
# declare -a scale=("12" "12" "12")
# declare -a ckpts=("9u1zyfqn" "ic7xh4va" "foyrso9f")

declare -a seeds=("1" "2" "3")
declare -a scale=("27" "27" "27")
declare -a ckpts=("hpalzipg" "ixtpm3oh" "7ht58n3r")

for i in "${!labels[@]}"
do
    for j in "${!scale[@]}"
    do
        sbatch scripts/submit_short_128G.sh train_rep.py --config configs/neurips/pnpl/s4l/camcan_${labels[i]}/${task[i]}.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ckpts[j]} --name neurips_camcan_s4l+${labels[i]}_${scale[j]}_task_${task[i]}_seed${seeds[j]}
    done
done