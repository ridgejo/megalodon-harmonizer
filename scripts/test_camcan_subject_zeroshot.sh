#! /bin/bash

declare -a ckpts=("oqrms1av" "nxmlpg0w" "g5z9sfth" "yoz3dt80" "0uvfxdfi" "ya0lfmtz" "xkafozc6" "ij0z3cxf" "9l2y3og7" "gnqz6ox7" "4h2dg35s" "acy2rs4g" "8af6wxr7" "po5qp79h" "uar8x3ak" "bwdrk3yv" "rfo81xud" "w6tc09ru" "u2mn8zf0" "7lzzi7h4" "ws5cnycw" "d6n5dpmc" "rcx151o5" "q6ecb66m" "nhgmouhx" "higj2ni0" "a1dt8m3z" "9oxmdq74" "yeli8c4z" "anyr9e8a")
declare -a seeds=("1" "2" "3" "1" "2" "3" "1" "2" "3" "1" "2" "3" "1" "2" "3" "1" "2" "3" "1" "2" "3" "1" "2" "3" "1" "2" "3" "1" "2" "3")
declare -a scale=("641" "641" "641" "312" "312" "312" "152" "152" "152" "74" "74" "74" "36" "36" "36" "17" "17" "17" "8" "8" "8" "4" "4" "4" "2" "2" "2" "1" "1" "1")

for i in "${!ckpts[@]}"
do

    # # Gwilliams voicing
    # sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/scaling/camcan/subject/gwilliams_voicing_zeroshot.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ckpts[i]} --name neurips_camcan_subject_zeroshot_${scale[i]}_gwilliams_voicing_seed${seeds[i]}

    # Gwilliams speech
    sbatch scripts/submit_short_256G.sh train_rep.py --config configs/neurips/pnpl/scaling/camcan/subject/gwilliams_speech_zeroshot.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ckpts[i]} --name neurips_camcan_subject_zeroshot_${scale[i]}_gwilliams_speech_seed${seeds[i]}

done