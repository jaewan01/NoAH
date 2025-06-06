dataset=("cora_coauth")

for data in ${dataset[@]}
do
    python reindexing.py -target ${data}
done
