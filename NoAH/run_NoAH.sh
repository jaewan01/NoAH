dataset=("cora_coauth")
seed=1
epoch=500
lr=0.01
device="cuda:0"
wdegreeset=("0.0" "0.1" "0.01")
wsizeset=("0.0" "1.0" "2.0")

for data in ${dataset[@]}
do  
    n_batch_c=0
    n_batch_f=0
    n_batch_bip=0
    
    elif [[ "$data" == "cora_coauth" ]]; then
    iter=50
    n_batch_f=2
    n_batch_bip=2

    elif [[ "$data" == "citeseer_cite" ]]; then
    iter=30
    n_batch_f=2
    n_batch_bip=2

    fi

    for wd in ${wdegreeset[@]}
    do
        for ws in ${wsizeset[@]}
        do
            python main.py -target ${data} -iter ${iter} -epoch ${epoch} -lr_c ${lr} -lr_f ${lr} -w_d ${wd} -w_s ${ws} -seed ${seed} -n_batch_c ${n_batch_c} -n_batch_f ${n_batch_f} -device ${device} -mode "NoAH"
            python main.py -target ${data} -iter ${iter} -epoch ${epoch} -lr_c ${lr} -w_d ${wd} -w_s ${ws} -seed ${seed} -n_batch_c ${n_batch_bip} -device ${device} -mode "NoAH_CF"
        done
    done
done
