#! /bin/bash

declare -a allckpts=("2up4w3u4" "xnegk86q" "15ld22le")
declare -a ampckpts=("g281x1cc" "k3ds0whk" "j66ua44g")
declare -a phaseckpts=()
declare -a bandckpts=("zyyf8ekr" "97p4elhd" "kco31932")
declare -a seeds=("1" "2" "3")

for i in "${!ampckpts[@]}"
do

    # All
    # # Armeni voicing
    # sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/ablate/camcan/shallow/all_armeni_voicing.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${allckpts[i]} --name neurips_camcan_ablate_all_armeni_voicing_seed${seeds[i]} --seed ${seeds[i]}
    # # Gwilliams voicing
    # sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/ablate/camcan/shallow/all_gwilliams_voicing.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${allckpts[i]} --name neurips_camcan_ablate_all_gwilliams_voicing_seed${seeds[i]} --seed ${seeds[i]}
    # # Armeni speech
    # sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/ablate/camcan/shallow/all_armeni_speech.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${allckpts[i]} --name neurips_camcan_ablate_all_armeni_speech_seed${seeds[i]} --seed ${seeds[i]}
    # # Gwilliams speech
    # sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/ablate/camcan/shallow/all_gwilliams_speech.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${allckpts[i]} --name neurips_camcan_ablate_all_gwilliams_speech_seed${seeds[i]} --seed ${seeds[i]}


    # Amplitude
    # # Armeni voicing
    # sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/ablate/camcan/shallow/amp_voicing.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ampckpts[i]} --name neurips_camcan_ablate_amp_armeni_voicing_seed${seeds[i]} --seed ${seeds[i]}
    # # Gwilliams voicing
    # sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/ablate/camcan/shallow/gwi_amp_voicing.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ampckpts[i]} --name neurips_camcan_ablate_amp_gwilliams_voicing_seed${seeds[i]} --seed ${seeds[i]}
    # # Armeni speech
    # sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/ablate/camcan/shallow/amp_speech.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ampckpts[i]} --name neurips_camcan_ablate_amp_armeni_speech_seed${seeds[i]} --seed ${seeds[i]}
    # # Gwilliams speech
    # sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/ablate/camcan/shallow/gwi_amp_speech.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ampckpts[i]} --name neurips_camcan_ablate_amp_gwilliams_speech_seed${seeds[i]} --seed ${seeds[i]}

    # Phase

    # Band
    # Armeni voicing
    sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/ablate/camcan/shallow/band_voicing.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${bandckpts[i]} --name neurips_camcan_ablate_band_armeni_voicing_seed${seeds[i]} --seed ${seeds[i]}
    # Gwilliams voicing
    sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/ablate/camcan/shallow/gwi_band_voicing.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${bandckpts[i]} --name neurips_camcan_ablate_band_gwilliams_voicing_seed${seeds[i]} --seed ${seeds[i]}
    # Armeni speech
    sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/ablate/camcan/shallow/band_speech.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${bandckpts[i]} --name neurips_camcan_ablate_band_armeni_speech_seed${seeds[i]} --seed ${seeds[i]}
    # Gwilliams speech
    sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/ablate/camcan/shallow/gwi_band_speech.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${bandckpts[i]} --name neurips_camcan_ablate_band_gwilliams_speech_seed${seeds[i]} --seed ${seeds[i]}

done