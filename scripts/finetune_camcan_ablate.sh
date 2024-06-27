#! /bin/bash

declare -a ampckpts=("g281x1cc" "k3ds0whk" "j66ua44g")
declare -a phaseckpts=()
declare -a bandckpts=()
declare -a seeds=("1" "2" "3")

for i in "${!ampckpts[@]}"
do

    # Amplitude
    # Armeni voicing
    sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/ablate/camcan/shallow/amp_voicing.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ampckpts[i]} --name neurips_camcan_ablate_amp_armeni_voicing_seed${seeds[i]} --seed ${seeds[i]}
    # Gwilliams voicing
    sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/ablate/camcan/shallow/gwi_amp_voicing.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ampckpts[i]} --name neurips_camcan_ablate_amp_gwilliams_voicing_seed${seeds[i]} --seed ${seeds[i]}
    # Armeni speech
    sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/ablate/camcan/shallow/amp_speech.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ampckpts[i]} --name neurips_camcan_ablate_amp_armeni_speech_seed${seeds[i]} --seed ${seeds[i]}
    # Gwilliams speech
    sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/ablate/camcan/shallow/gwi_amp_speech.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ampckpts[i]} --name neurips_camcan_ablate_amp_gwilliams_speech_seed${seeds[i]} --seed ${seeds[i]}

    # Phase

    # Band

done