# __NoAH__
This is the official implementation of __NoAH__ (Node Attribute based Hypergraph generator), which is described in the following papers:
* __Attributed Hypergraph Generation with Realistic Interplay Between Structure and Attributes__ (http://arxiv.org/abs/2509.21838)
<br> Jaewan Chun\*, Seokbum Yoon\*, Minyoung Choe, Geon Lee, Kijung Shin
<br> ICDM 2025 

* __From binary to general attributes: attributed hypergraph generation with realistic interplay between structure and attributes__ (https://link.springer.com/article/10.1007/s10115-026-02850-x)
<br> Jaewan Chun, Seokbum Yoon, Minyoung Choe, Geon Lee, Kijung Shin
<br> Knowledge and Information Systems

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
