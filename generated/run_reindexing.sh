# dataset=("contact_highschool" "contact_workspace")
# dataset=("reviews_bluesmusic" "reviews_madisonrestaurant" "reviews_vegasbar")
# dataset=("citeseer_cite" "cora_coauth")
# dataset=("devops_stack" "patents_stack")
# dataset=("pubmed_cite")

models=("noah" "noah_cf") # original NoAH and NoAH with core-fringe affinity matrix
models=("noah_x" "noah_x_plus") # NoAH without node attributes
models=("noah_categorical") # NoAH with categorical node attributes
models=("noah_continuous") # NoAH with continuous node attributes

for data in ${dataset[@]}
do
    python reindexing.py -target ${data} --models ${models[@]}
done
