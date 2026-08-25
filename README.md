# __NoAH__
This is the official implementation of __NoAH__ (Node Attribute based Hypergraph generator), which is described in the following papers:
* __Attributed Hypergraph Generation with Realistic Interplay Between Structure and Attributes__ (http://arxiv.org/abs/2509.21838)
<br> Jaewan Chun\*, Seokbum Yoon\*, Minyoung Choe, Geon Lee, Kijung Shin
<br> ICDM 2025 

* __From binary to general attributes: attributed hypergraph generation with realistic interplay between structure and attributes__ (https://link.springer.com/article/10.1007/s10115-026-02850-x)
<br> Jaewan Chun, Seokbum Yoon, Minyoung Choe, Geon Lee, Kijung Shin
<br> Knowledge and Information Systems, 2026

## __Overview__
In many real-world scenarios, interactions happen in a group-wise manner with multiple entities, and therefore, hypergraphs are a suitable tool to accurately represent such interactions. 
Hyperedges in real-world hypergraphs are not composed of randomly selected nodes but are instead formed through structured processes. 
Consequently, various hypergraph generative models have been proposed to explore fundamental mechanisms underlying hyperedge formation. 
However, most existing hypergraph generative models do not account for node attributes, which can play a significant role in hyperedge formation. 
As a result, these models fail to reflect the interactions between structure and node attributes. 

To address the issue above, we propose NoAH, a stochastic hypergraph generative model for attributed hypergraphs. 
NoAH utilizes the core–fringe node hierarchy to model hyperedge formation as a series of node attachments and determines attachment probabilities based on node attributes. 
We further introduce NoAHFit, a parameter learning procedure that fits NoAH to a given real-world hypergraph so that generated hypergraphs reproduce structural and attribute-related patterns. 
Through experiments on nine datasets across four different domains, we show that NoAH with NoAHFit achieves the best overall average rank among the nine evaluated hypergraph generative models when evaluated across six structure–attribute interplay metrics.
Moreover, we discuss variants of NoAH for different types of node attributes, including binary, categorical, and continuous attributes. 
For cases without pre-existing node attributes, we extend NoAH and NoAHFit to jointly learn latent node attributes together with the parameters of NoAH and use the learned attributes for generation.

## Repository layout

| Directory | Purpose | Attributes |
|---|---|---|
| `NoAH/` | NoAH, NoAHFit, and core–fringe ablations | Binary |
| `NoAH_categorical/` | Generalized categorical affinities | Categorical |
| `NoAH_continuous/` | Bilinear and neural affinity variants | Continuous |
| `NoAH_X/` | NoAHFit-X, NoAH-X, and NoAH-X+ | None provided |
| `metric/` | Structure–attribute interplay evaluation | Binary |
| `dataset/` | Input hypergraphs and attributes | — |
| `generated/` | Stroing generated hypergraphs and reindexing utility | — |

## Running the models

The shell scripts contain dataset-specific defaults for recovery iterations and fitting batches. Edit the configuration block at the top of a script before running a full experiment.

### Binary attributes

```bash
cd NoAH
bash run_NoAH.sh
```

The script runs the proposed model and two ablations over degree- and cardinality-loss weight grids:

- `NoAH`: proposed method with a UMHS-based core–fringe split.
- `NoAH-dCF`: same model with degree-based core selection.
- `NoAH_noCF`: no core–fringe distinction.

### Categorical attributes

```bash
cd NoAH_categorical
bash run_NoAH_categorical.sh
```

This reads `attribute_categorical.txt` and `attribute_categorical_counts.txt`.

### Continuous attributes

```bash
cd NoAH_continuous
bash run_NoAH_continuous.sh  # bilinear affinities
bash run_NoAH_neural.sh      # neural affinities
```

These use the PubMed TF-IDF vectors in `dataset/pubmed_cite/attribute_raw.txt`. 

### No pre-existing attributes

```bash
cd NoAH_X
bash run_NoAH_X.sh
```

One run produces outputs for both NoAH-X and NoAH-X+. Set `k` in the script, or pass `-k`, to choose the latent dimension. 

## Data format

Store each dataset in `dataset/<name>/`. Node IDs must be zero-based contiguous integers, as they index attribute rows.

- `hyperedge.txt`: one comma-separated hyperedge per line, e.g. `0,1,4`.
- `attribute.txt`: one comma-separated binary vector per node, in node-ID order.
- `attribute_categorical.txt`: one integer-valued vector per node.
- `attribute_categorical_counts.txt`: category counts by dimension, e.g. `3,2,4`.
- `attribute_raw.txt`: one real-valued vector per node; the continuous loader normalizes each dimension to `[0,1]`.

Only the files required by the selected model need to be present.

## Outputs

Generated hypergraphs are written to `generated/<model>/<dataset>/` as `.txt` files with one comma-separated hyperedge per line. Parameters and seeds are encoded in filenames. Attribute-aware reindexing may also create an `-indices.txt` mapping from generated row indices to original node IDs.

The binary implementation reindexes automatically. For other `*-preindexing.txt` outputs, configure and run:

```bash
cd generated
bash run_reindexing.sh
```

## __Datasets__
We provide the code for NoAH. We provide the information on the datasets used in the experiment below.

|Dataset|Cores|Fringes|Nodes|Hyperedges|Attribute Dimension|
|:---:|:---:|:---:|:---:|:---:|:---:|
|[Citeseer](https://github.com/malllabiisc/HyperGCN)|597|861|1,458|1,079|3,703|
|[Cora](https://github.com/malllabiisc/HyperGCN)|841|1,547|2,388|1,072|1,433|
|[High School](http://www.sociopatterns.org/datasets/)|288|39|327|7,818|12|
|[Workspace](http://www.sociopatterns.org/datasets/)|71|21|92|788|5|
|[Amazon Music](https://jmcauley.ucsd.edu/data/amazon/)|379|727|1,106|686|7|
|[Yelp Resaurant](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset)|273|292|565|594|9|
|[Yelp Bar](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset)|625|609|1,234|1,188|15|
|[Devops](https://archive.org/download/stackexchange)|2,003|3,007|5,010|5,684|429|
|[Patents](https://archive.org/download/stackexchange)|894|3,564|4,458|4,669|2,170|

## __Requirements__

__NoAH__ and the evaluation codes were run with the following Python packages:

| Package         | Version   |
|----------------|-----------|
| `networkx`      | 3.1       |
| `numpy`         | 1.26.4    |
| `scikit-learn`  | 1.3.2     |
| `scipy`         | 1.10.1    |
| `snap-stanford` | 6.0.0     |
| `torch`         | 2.3.0     |
| `tqdm`          | 4.65.0    |

You can install them via pip:

```
pip install -r requirements.txt
```

## __Citation__
If you find this work useful, please consider citing:
```
@INPROCEEDINGS{11391927,
  author={Chun, Jaewan and Yoon, Seokbum and Choe, Minyoung and Lee, Geon and Shin, Kijung},
  booktitle={ICDM}, 
  title={Attributed Hypergraph Generation with Realistic Interplay Between Structure and Attributes}, 
  year={2025},
  pages={189-198},
  doi={10.1109/ICDM65498.2025.00026}
}

@article{chun2026binary,
  title={From binary to general attributes: attributed hypergraph generation with realistic interplay between structure and attributes},
  author={Chun, Jaewan and Yoon, Seokbum and Choe, Minyoung and Lee, Geon and Shin, Kijung},
  journal={Knowledge and Information Systems},
  volume={68},
  number={1},
  pages={245},
  year={2026},
  publisher={Springer}
}
```
