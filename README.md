# GraphSimEmbed

For graphs with many classes (thousands or millions) of node and only a few examples per class, 
cross entropy classification loss is impractical. Instead, we can output a fixed dimension embedding and
classify or cluster nodes by the pairwise distance between them in the embedding space. 

## Files

* GCNSimEmbed.py uses a graph convolutional network trained with triplet loss based on either MSE or cosine distance. 
Cannot scale since it requires the full graph adjacency matrix for each training step.
* SageSimEmbed.py uses [GraphSage](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) stochastic minibatch training method to scale to arbitrarily large graphs. Adapted from [DGL reference implementation](https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage).  
* geosim.py simulates graph data with arbitrary number of classes and a random number of examples per class between 1-3.
* SageAPI.py implements an API for faiss index based on similarity embeddings

To run this code, all that is needed is docker (or nvidia docker for gpu support). Clone the repo and run the following commands.

* mkdir ./chunks
* mkdir ./ims
* docker build -t dgl .
* ./dockerrun.sh (you can change where the simulated data will be stored by changing which directory is mounted to '/geosim' in the container)

## Results

First we need to simulate some data

```bash
python3 ./sse/geosim.py --nfiles 100 --npaths 1000
```

Then we can build a graph, and generate embeddings for each node.

```bash
python3 ./sse/SageSimEmbed.py \
--num-epochs 100 --batch-size 1000 \
--test-every 10 --lr .01 --n-classes 100000 \
--p-train 0.01 --distance-metric cosine --embedding-dim 512 --num-hidden 512 \
--sup-weight 0.5 --neg_samples 1 --agg-type gcn --device cpu --save
```

The untrained model achieves 95.9% top1 accuracy and 99.4% top5 accuracy vs 75.2% and 93.2% respectively from counting mutual neighbors. The performance does not improve with training unless a substantial fraction of node labels are known at training time (e.g. the training is mostly supervised).

## Recommender API

Once we have an embedding for all nodes, we can train a faiss index to quickly search for nearest neighbors. Then, we can query the faiss index via an API to get reccomendations for similar nodes. 

To test on localhost, you can leave all args as default.

```bash
uvicorn SageAPI:app
```

Then simply send a request using the id of the node set by geosim.py.

```python
import requests
#configure url, default is localhost
api = 'http://127.0.0.1:8000/{}/{}/{}'
#this is samplenumber_hash and is found in the simulated data csv files
example_node = '0_8424d2fa9b82652458a7da8483afc6fa5a3963010cc543bfa3ed3f7e1f0ed577'
#how many neighbors to find
k = 5
r = requests.get(apiurl.format('knn',nodeid,k))
r
{'Nearest Neighbors': ['0_8424d2fa9b82652458a7da8483afc6fa5a3963010cc543bfa3ed3f7e1f0ed577',
                       '1_8424d2fa9b82652458a7da8483afc6fa5a3963010cc543bfa3ed3f7e1f0ed577',
                       '2_8424d2fa9b82652458a7da8483afc6fa5a3963010cc543bfa3ed3f7e1f0ed577',
                       '1_f6104461652a1ca6528a0696a1b2cf211cfcb658b8f60e1a3f753b5c73ee4434',
                       '2_a85539018d2d4365f819f0b102991fbf74132e933a596cf542b190ba3df0d194'],
 'QueryID': '0_8424d2fa9b82652458a7da8483afc6fa5a3963010cc543bfa3ed3f7e1f0ed577',
 'Score': [0.0,
           0.03939133882522583,
           0.0400189608335495,
           0.1584293395280838,
           0.1703648567199707]}

#we can also query neighbors based on number of mutual neighbors as a baseline
r = requests.get(apiurl.format('knn_jacard',nodeid,k))
r
{...
 'N Shared Locations': [38, 30, 21, 21, 20],
 'Nearest Neighbors': ['0_8424d2fa9b82652458a7da8483afc6fa5a3963010cc543bfa3ed3f7e1f0ed577',
                       '2_8424d2fa9b82652458a7da8483afc6fa5a3963010cc543bfa3ed3f7e1f0ed577',
                       '0_360dea129cf0cccb5ce1d590b2217d589de0c4d5e547cbfa5b4478284f939d3d',
                       '2_c9ad60ed87d3d334bbe7d7a2ea68a7c3c27c76c7ea43cb373bc71a96568abb92',
                       '1_8424d2fa9b82652458a7da8483afc6fa5a3963010cc543bfa3ed3f7e1f0ed577'],
 'QueryID': '0_8424d2fa9b82652458a7da8483afc6fa5a3963010cc543bfa3ed3f7e1f0ed577'}

```