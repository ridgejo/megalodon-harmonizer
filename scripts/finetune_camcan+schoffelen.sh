#! /bin/bash

declare -a ckpts=("z97s7965")
declare -a seeds=("3")

for i in "${!ckpts[@]}"
do

    # Armeni voicing
    sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/scaling/camcan/sch_armeni_voicing.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ckpts[i]} --name neurips_camcan+schoffelen_armeni_voicing_seed${seeds[i]} --seed ${seeds[i]}

    # Gwilliams voicing
    sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/scaling/camcan/sch_gwilliams_voicing.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ckpts[i]} --name neurips_camcan+schoffelen_gwilliams_voicing_seed${seeds[i]} --seed ${seeds[i]}

    # Armeni speech
    sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/scaling/camcan/sch_armeni_speech.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ckpts[i]} --name neurips_camcan+schoffelen_armeni_speech_seed${seeds[i]} --seed ${seeds[i]}

    # Gwilliams speech
    sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/scaling/camcan/sch_gwilliams_speech.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ckpts[i]} --name neurips_camcan+schoffelen_gwilliams_speech_seed${seeds[i]} --seed ${seeds[i]}

done