#! /bin/bash

# declare -a ckpts=("uahsvoaw" "lwxbsduq" "4mx0hhs3")
# declare -a seeds=("1" "2" "3")
# declare -a scale=("schmax" "schmax" "schmax")


declare -a ckpts=("4mx0hhs3")
declare -a seeds=("3")
declare -a scale=("schmax")

for i in "${!ckpts[@]}"
do

    # Armeni voicing
    sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/scaling/camcan/sch_armeni_voicing.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ckpts[i]} --name neurips_${scale[i]}_armeni_voicing_seed${seeds[i]}

    # Gwilliams voicing
    sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/scaling/camcan/sch_gwilliams_voicing.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ckpts[i]} --name neurips_${scale[i]}_gwilliams_voicing_seed${seeds[i]}

    # Armeni speech
    sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/scaling/camcan/sch_armeni_speech.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ckpts[i]} --name neurips_${scale[i]}_armeni_speech_seed${seeds[i]}

    # Gwilliams speech
    sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/scaling/camcan/sch_gwilliams_speech.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ckpts[i]} --name neurips_${scale[i]}_gwilliams_speech_seed${seeds[i]}

done