dataset=("cora_coauth")

seed=1
epoch=1
lr=0.01
wdegreeset=("0.0" "0.1" "0.01")
wsizeset=("0.0" "1.0" "2.0")

for data in ${dataset[@]}
do
    if [ "$data" == "cora_coauth" ]; then
    iter=50
    
    elif [ "$data" == "citeseer_cite" ]; then
    iter=30

    fi

    mkdir -p results/answer/${data}
    python structure_attribute_interplay.py --inputpath ../dataset/${data}/hyperedge --outputdir results/answer/${data}/ -target ${data} 

    for wd in ${wdegreeset[@]}
    do
        for ws in ${wsizeset[@]}
        do
            mkdir -p results/NoAH/${data}/${iter}-${lr}-${lr}-${wd}-${ws}-${epoch}-${seed}
            python structure_attribute_interplay.py --inputpath ../generated/NoAH/${data}/NoAH-${iter}-${lr}-${lr}-${wd}-${ws}-${epoch}-${seed} --outputdir results/NoAH/${data}/${iter}-${lr}-${lr}-${wd}-${ws}-${epoch}-${seed}/ -target ${data} 

            mkdir -p results/NoAH_CF/${data}/${iter}-${lr}-${lr}-${wd}-${ws}-${epoch}-${seed}
            python structure_attribute_interplay.py --inputpath ../generated/NoAH_CF/${data}/NoAH_CF-${iter}-${lr}-${lr}-${wd}-${ws}-${epoch}-${seed} --outputdir results/NoAH_CF/${data}/${iter}-${lr}-${lr}-${wd}-${ws}-${epoch}-${seed}/ -target ${data}
        done
    done

done

models=("NoAH" "NoAH_CF")
dataset=("cora_coauth")

for data in ${dataset[@]}
do

    for model in ${models[@]}
    do
        python ablation.py --dataname ${data} --ablation_target ${model} 
    done

    python gen_table.py --dataname ${data} 

done

