#! /bin/bash

declare -a ckpts=("2up4w3u4" "xnegk86q" "15ld22le")
declare -a seeds=("1" "2" "3")
declare -a scale=("641" "641" "641")

for i in "${!ckpts[@]}"
do

    # Armeni voicing
    sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/scaling/camcan/supervised/armeni_voicing.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ckpts[i]} --name neurips_camcan_pretrain_${scale[i]}_armeni_voicing_seed${seeds[i]}

    # Gwilliams voicing
    sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/scaling/camcan/supervised/gwilliams_voicing.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ckpts[i]} --name neurips_camcan_pretrain_${scale[i]}_gwilliams_voicing_seed${seeds[i]}

    # Armeni speech
    sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/scaling/camcan/supervised/armeni_speech.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ckpts[i]} --name neurips_camcan_pretrain_${scale[i]}_armeni_speech_seed${seeds[i]}

    # Gwilliams speech
    sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/scaling/camcan/supervised/gwilliams_speech.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ckpts[i]} --name neurips_camcan_pretrain_${scale[i]}_gwilliams_speech_seed${seeds[i]}

done