#! /bin/bash

declare -a ckpts=("h4dt9ivg" "dupc3itu" "8zc5f59g" "l3uwnpf3" "l892i29r" "yjzscdqa" "m5r48ajg" "gbhavgs1" "wjvlyohx" "i0u1pnfv" "5ms2uh82" "5yqng045")
declare -a seeds=("87" "93" "52" "87" "93" "52" "87" "93" "52" "87" "93" "52")
declare -a scale=("10" "10" "10" "20" "20" "20" "40" "40" "40" "80" "80" "80")

for i in "${!ckpts[@]}"
do
    sbatch scripts/submit.sh train_rep.py --config configs/neurips/pnpl/scaling/armeni_001/gwi_speech.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ckpts[i]} --name neurips_arm001_${scale[i]}_gwi_speech_seed_${seeds[i]}

    sbatch scripts/submit.sh train_rep.py --config configs/neurips/pnpl/scaling/armeni_001/gwi_voicing.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ckpts[i]} --name neurips_arm001_${scale[i]}_gwi_voicing_seed_${seeds[i]}
done