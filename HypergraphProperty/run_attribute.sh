dataset=("cora_coauth")

pset_ff=("0.42" "0.45" "0.48")
qset_ff=("0.1" "0.2" "0.3")

bset=("8" "12" "15")
pset=("0.5" "0.7" "0.9")
cset=("2.0" "6.0" "10.0")

seed=1
epoch=500
lr=0.01
wdegreeset=("0.0" "0.1" "0.01")
wsizeset=("0.0" "1.0" "2.0")

wdegreeset_rec=("0.0000" "0.0100" "0.1000")
wsizeset_rec=("0.0000" "1.0000" "2.0000")
lr_rec=0.010
an=0.00005
unit=2

gammaset=("0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")

dk_set=("0-0" "0-1" "1-0")

for data in ${dataset[@]}
do
    if [ "$data" == "cora_coauth" ]; then
    iter=50
    x=7
    y=6
    z=4
    
    elif [ "$data" == "citeseer_cite" ]; then
    iter=30
    x=12
    y=11
    z=3

    elif [ "$data" == "devops_stack" ]; then
    iter=30
    x=18
    y=18
    z=3
    wdegreeset=("10.0" "100.0" "1000.0")
    wsizeset=("10.0" "100.0" "1000.0")
    wdegreeset_rec=("10.0000" "100.0000" "1000.0000")
    wsizeset_rec=("10.0000" "100.0000" "1000.0000")

    elif [ "$data" == "patents_stack" ]; then
    iter=30
    x=17
    y=17
    z=3
    wdegreeset=("10.0" "100.0" "1000.0")
    wsizeset=("10.0" "100.0" "1000.0")
    wdegreeset_rec=("10.0000" "100.0000" "1000.0000")
    wsizeset_rec=("10.0000" "100.0000" "1000.0000")


    elif [[ "$data" == "reviews_bluesmusic" || "$data" == "reviews_madisonrestaurant" || "$data" == "reviews_vegasbar" ]]; then
    iter=30

    elif [[ "$data" == "contact_workspace" || "$data" == "contact_highschool" ]]; then
    iter=1

    fi

    # mkdir -p results/answer/${data}
    # python src/attribute_related.py --inputpath ../dataset/${data}/hyperedge --outputdir results/answer/${data}/ -target ${data} 

    # mkdir -p results/cl/${data}
    # python src/attribute_related.py --inputpath ../generated/cl/${data}/cl --outputdir results/cl/${data}/ -target ${data} 

    # for p in ${pset_ff[@]}
    # do
    #     for q in ${qset_ff[@]}
    #     do
    #         mkdir -p results/ff/${data}/${p}-${q}
    #         python src/attribute_related.py --inputpath ../generated/ff/${data}/ff-${p}-${q} --outputdir results/ff/${data}/${p}-${q}/ -target ${data}
    #     done
    # done

    # mkdir -p results/lap/${data}
    # python src/attribute_related.py --inputpath ../generated/lap/${data}/lap --outputdir results/lap/${data}/ -target ${data}

    # mkdir -p results/pa/${data}
    # python src/attribute_related.py --inputpath ../generated/pa/${data}/pa --outputdir results/pa/${data}/ -target ${data}

    # for b in ${bset[@]}
    # do
    #     for p in ${pset[@]}
    #     do 
    #         for c in ${cset[@]}
    #         do
    #             mkdir -p results/tr/${data}/${b}-${p}-${c}
    #             python src/attribute_related.py --inputpath ../generated/tr/${data}/tr-${b}-${p}-${c} --outputdir results/tr/${data}/${b}-${p}-${c}/ -target ${data} 
    #         done
    #     done
    # done

    # for wd in ${wdegreeset_rec[@]}
    # do
    #     for ws in ${wsizeset_rec[@]}
    #     do
    #         mkdir -p results/hyrec/${data}/${x}-${y}-${z}-${lr_rec}-${unit}-${an}-sl${ws}-dl${wd}
    #         python src/attribute_related.py --inputpath ../generated/hyrec/${data}/hyrec-${x}-${y}-${z}-${lr_rec}-${unit}-${an}-sl${ws}-dl${wd} --outputdir results/hyrec/${data}/${x}-${y}-${z}-${lr_rec}-${unit}-${an}-sl${ws}-dl${wd}/ -target ${data}
    #     done
    # done

    # for gamma in ${gammaset[@]}
    # do
    #     mkdir -p results/sbm/${data}/${gamma}
    #     python src/attribute_related.py --inputpath ../generated/sbm/${data}/sbm-${gamma} --outputdir results/sbm/${data}/${gamma}/ -target ${data}
    # done

    # for dk in ${dk_set[@]}
    # do
    #     mkdir -p results/dk/${data}/${dk}
    #     python src/attribute_related.py --inputpath ../generated/dk/${data}/dk-${dk} --outputdir results/dk/${data}/${dk}/ -target ${data}
    # done

    for wd in ${wdegreeset[@]}
    do
        for ws in ${wsizeset[@]}
        do
            mkdir -p results/noah/${data}/${iter}-${lr}-${lr}-${wd}-${ws}-${epoch}-${seed}
            python src/attribute_related.py --inputpath ../generated/noah/${data}/noah-${iter}-${lr}-${lr}-${wd}-${ws}-${epoch}-${seed} --outputdir results/noah/${data}/${iter}-${lr}-${lr}-${wd}-${ws}-${epoch}-${seed}/ -target ${data} 

            mkdir -p results/bi/${data}/${iter}-${lr}-${lr}-${wd}-${ws}-${epoch}-${seed}
            python src/attribute_related.py --inputpath ../generated/bi/${data}/bi-${iter}-${lr}-${lr}-${wd}-${ws}-${epoch}-${seed} --outputdir results/bi/${data}/${iter}-${lr}-${lr}-${wd}-${ws}-${epoch}-${seed}/ -target ${data}
        done
    done

done
