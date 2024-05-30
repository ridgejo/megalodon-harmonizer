#! /bin/bash

# declare -a ckpts=("xmgfi00u" "dwyd3zb1" "udehz91m")
# declare -a seeds=("1" "2" "3")
# declare -a scale=("36" "36" "36")

# declare -a ckpts=("bp3l9ibx" "4p1wfzsl" "h7q9ji7r")
# declare -a seeds=("1" "2" "3")
# declare -a scale=("74" "74" "74")

# declare -a ckpts=("fqhrv89e" "6ohu7z3z" "70v3p1k4")
# declare -a seeds=("1" "2" "3")
# declare -a scale=("1" "1" "1")

# declare -a ckpts=("51p4a32z" "4pefphb1" "k2ylqgzn")
# declare -a seeds=("1" "2" "3")
# declare -a scale=("2" "2" "2")

# declare -a ckpts=("6pu0i4j5" "uk61qmea" "q19398qy")
# declare -a seeds=("1" "2" "3")
# declare -a scale=("4" "4" "4")

# declare -a ckpts=("ca9lf1li" "3cch5az0" "h6wf9tk9")
# declare -a seeds=("1" "2" "3")
# declare -a scale=("8" "8" "8")

# declare -a ckpts=("bdy5w2by" "6h7exixt" "vaffdoxw")
# declare -a seeds=("1" "2" "3")
# declare -a scale=("17" "17" "17")

# declare -a ckpts=("7iu4r8pb" "b0h86kcj" "gobfg8f8")
# declare -a seeds=("1" "2" "3")
# declare -a scale=("152" "152" "152")

declare -a ckpts=("miokw8n6" "w1s7rdps" "jveuvcwt")
declare -a seeds=("1" "2" "3")
declare -a scale=("312" "312" "312")

for i in "${!ckpts[@]}"
do

    # Armeni voicing
    sbatch scripts/submit_short_128G.sh train_rep.py --config configs/neurips/pnpl/scaling/camcan/armeni_voicing.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ckpts[i]} --name neurips_camcan_${scale[i]}_armeni_voicing_seed${seeds[i]}

    # Gwilliams voicing
    sbatch scripts/submit_short_128G.sh train_rep.py --config configs/neurips/pnpl/scaling/camcan/gwilliams_voicing.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ckpts[i]} --name neurips_camcan_${scale[i]}_gwilliams_voicing_seed${seeds[i]}

    # Armeni speech
    sbatch scripts/submit_short_128G.sh train_rep.py --config configs/neurips/pnpl/scaling/camcan/armeni_speech.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ckpts[i]} --name neurips_camcan_${scale[i]}_armeni_speech_seed${seeds[i]}

    # Gwilliams speech
    sbatch scripts/submit_short_128G.sh train_rep.py --config configs/neurips/pnpl/scaling/camcan/gwilliams_speech.yaml --checkpoint /data/engs-pnpl/lina4368/experiments/MEGalodon-representation/${ckpts[i]} --name neurips_camcan_${scale[i]}_gwilliams_speech_seed${seeds[i]}

done