# __NoAH__
This is the official implementation of __NoAH__ (Node Attribute based Hypergraph generator), which is described in the following paper:
* __Attributed Hypergraph Generation with Realistic Interplay Between Structure and Attributes__

## __Overview__
In many real-world scenarios, interactions happen in a group-wise manner with multiple entities, and therefore, hypergraphs are a suitable tool to accurately represent such interactions. 
Hyperedges in real-world hypergraphs are not composed of randomly selected nodes but are instead formed through structured processes. 
Consequently, various hypergraph generative models have been proposed to explore fundamental mechanisms underlying hyperedge formation. 
However, most existing hypergraph generative models do not account for node attributes, which can play a significant role in hyperedge formation.
As a result, these models fail to reflect the interactions between structure and node attributes.
<br>
To address the issue above, we propose NOAH, a stochastic hypergraph generative model for attributed hypergraphs. 
NOAH utilizes the core–fringe node hierarchy to model hyperedge formation as a series of node attachments and determines attachment probabilities based on node attributes. 
We further introduce NOAHFIT, a parameter learning procedure that allows NOAH to replicate a given real-world hypergraph. 
Through experiments on nine datasets across four different domains, we show that NOAH with NOAHFIT more accurately reproduces the structure–attribute interplay observed in the real-world hypergraphs than eight baseline hypergraph generative models, in terms of six metrics.

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

## __How to Run NoAH__
1. Run __NoAH/run_NoAH.sh__, with designated configuration. 
In particular, you can adjust (1) dataset, (2) seed (random seed), (3) epoch, (4) lr (learning rate), (5) device (CUDA device), (6) wdegreeset (set of degree penalty weight), and (7) wsizeset (set of cardinality penalty weight).
2. Run __metric/run_metric.sh__.
