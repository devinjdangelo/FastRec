# FastRec

Graph neural networks are capable of capturing the structure and relationships between nodes in a graph as dense vectors. 
With these dense vectors, we can identify pairs of nodes that are similar, identify communities and clusters, or train
a linear classification model with the dense vectors as inputs. 

This project automates the entire pipeline from node/edge graph data to generate embeddings, train and fine tune those embeddings, create and train a [Facebook AI Similarity Search Index](https://ai.facebook.com/tools/faiss/) (faiss), and deploy a recommender API to query the index over the network. FastRec handles all of the boilerplate code, handling gpu/cpu memory management, and passing data between pytorch, Deep Graph Library (DGL), faiss, and fastapi. 

The code is intended to be as scalable as possible, with the only limitation being the memory available to store the graph. The code adapts the implementation of [GraphSage](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) from the [DGL reference implementation](https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage). FastRec has been tested on graphs with up to 1 million nodes and 100 million edges and was able to generate and train embeddings, train a faiss index, and begin answering api queries in minutes. With sufficient memory, it should be able to scale to billions of nodes and edges. Distributed training is not currently implemented, but could further improve scalability. 

## Installation

A dockerfile is included with all dependencies needed. Simply clone the repo, build the dockerfile, and run the code in the built image. Nvidia docker is needed for gpu support. There is not currently a pip or conda package.

## Basic Usage: Karate Club Communities

As an example, we can generate embeddings for [Zachary's karate club](https://en.wikipedia.org/wiki/Zachary%27s_karate_club) graph. See [karateclub.py](https://github.com/devinjdangelo/FastRec/blob/master/examples/karateclub.py) for the full script to replicate the below.

First, convert the graph into a node and edgelist format.

```python
import networkx as nx
g = nx.karate_club_graph()
nodes = list(g.nodes)
e1,e2 = zip(*g.edges)
attributes = pd.read_csv('./karate_attributes.csv')
```

Then we can initialize a recommender, add the data, and generate node embeddings.

```python
from fastrec import GraphRecommender
#initialize our recommender to embed into 2 dimensions and 
#use euclidan distance as the metric for similarity.
sage = GraphRecommender(2,distance='l2')
sage.add_nodes(nodes)
sage.add_edges(e1,e2)
sage.add_edges(e2,e1)
sage.update_labels(attributes.community)
untrained_embeddings =  sage.embeddings
```
How do the embeddings look? Even with no training of the graph neural network weights, the embeddings don't do a terrible job  dividing the two communities. The nodes in the Instructor community are blue and the nodes in the Administrator community are red.

<img src="https://github.com/devinjdangelo/FastRec/blob/master/examples/graphics/untrained_example_supervised.png" alt="drawing" width="600"/>

With one command, we can improve the embeddings with supervised learning with a triplet loss. 

```python
epochs, batch_size = 150, 15
sage.train(epochs, batch_size)
```
<img src="https://github.com/devinjdangelo/FastRec/blob/master/examples/graphics/supervised.gif" alt="drawing" width="600"/>

The trained embeddings much more neatly divide the communities. But what about the more realistic scenario where we did not know the labels of all of the nodes in advance? We can instead train the embeddings in a fully unsupervised manner.

```python
epochs, batch_size = 150, 15
sage.train(epochs, batch_size, unsupervised=True)
```

<img src="https://github.com/devinjdangelo/FastRec/blob/master/examples/graphics/unsupervised.gif" alt="drawing" width="600"/>

In this case, the unsupervised training actually seems to do a slightly better job of dividing the two communities.

What if we have a very large graph which is expensive and slow to train? Often, the untrained performance of the embeddings will improve if we increase the size of our graph neural network (in terms of width and # of parameters).  

```python
sage = GraphRecommender(2,distance='l2',feature_dim=512,hidden_dim=512)
untrained_embeddings_large = sage.embeddings
```

<img src="https://github.com/devinjdangelo/FastRec/blob/master/examples/graphics/untrained_example_large.png" alt="drawing" width="600"/>

This looks nearly as good as the trained version of the small network, but no training was required! 

Once we have embeddings that we are happy with, we can query a specific node or nodes to get its nearest neighbors in a single line.

```python
#what are the 5 nearest neighbors of node 0, the Admin, and 33, the Instructor?
sage.query_neighbors(['0','33'],k=5)
{'0': {'neighbors': ['0', '13', '16', '6', '5'], 'distances': [0.0, 0.001904212054796517, 0.005100540816783905, 0.007833012379705906, 0.008420777507126331]}, '33': {'neighbors': ['33', '27', '31', '28', '32'], 'distances': [0.0, 0.0005751167191192508, 0.0009900123113766313, 0.001961079193279147, 0.006331112235784531]}}
```
Each node's nearest neighbor is itself with a distance of 0. The Admin is closest to nodes 13, 16, 6, and 5, all of which are in fact part of the Admin community. The Instructor is closest to 27, 31, 28, and 32, all of which are part of the Instructor community. 

## Reddit Post Recommender

In under 5 minutes and with just 10 lines of code, we can create and deploy a Reddit post recommender based on a graph dataset with over 100m edges. We will use the Reddit post dataset from the [GraphSage](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) paper. Each node represets a post and an edge between posts represents one user who commented on both posts. Each node is labeled with one of 41 subreddits, which group the posts by theme or user interest. The original paper focused on correctly classifying the subreddit of each post. Here, we will simply say that a post recommendation is reasonable if it is in the same subreddit as the query post. See [reddit.py](https://github.com/devinjdangelo/FastRec/blob/master/examples/reddit.py) for the full script to replicate the below.

First, we download the Reddit Dataset. 

```python
import pandas as pd
import numpy as np
from dgl.data import RedditDataset
data = RedditDataset(self_loop=True)
e1, e2 = data.graph.all_edges()
e1, e2 = e1.numpy(), e2.numpy()
nodes = pd.DataFrame(data.labels,dtype=np.int32,columns=['labels'])
```

Now we can set up our recommender. For larger graphs, it will be much faster to use gpu for both torch and faiss computations.

```python
from fastrec import GraphRecommender
sage = GraphRecommender(128, feature_dim=512, hidden_dim=256, 
    torch_device='cuda', faiss_gpu=True, distance='cosine')
sage.add_nodes(nodes.index.to_numpy())
sage.add_edges(e1,e2)
sage.update_labels(nodes.labels)
```

Finally, we can evaluate our untrained embedding and deploy our API.

```python
perf = sage.evaluate(test_levels=[10,5])
print(perf)
{'Top 10 neighbors': {'Share >=1 correct neighbor': 0.9517867490824802, 'Share of correct neighbors': 0.8623741763784262}, 'Top 5 neighbors': {'Share >=1 correct neighbor': 0.9417079818856909, 'Share of correct neighbors': 0.8764973279247956}}
sage.start_api()
```

The performance stats indicate that on average 86% of the top 10 recommendations for a post are in the same subreddit. About 95% of all posts have at least 1 recommendation in the same subreddit among its top 10 recommendations. We could optionally train our embeddings with supervised or unsupervised learning from here, but for now this performance is good enough. We can now query our API over the network.

## Recommender API

We can share the recommender system as an API in a single line. No args are needed to test over localhost, but we can optionally pass in any args accepted by [uvicorn](https://www.uvicorn.org/deployment/).

```python
host, port = 127.0.0.1, 8000
sage.start_api(host=host,port=port)
```

This method of starting the API is convenient but has some downsides in the current implementation. Some data will be duplicated in memory, so if your graph is taking up most of your current memory this deployment may fail. You can avoid this issue by instead using the included deployment script. Simply save your GraphRecommender and point the deployment script to the saved location.

```bash
fastrec-deploy /example/directory
```

Now we can query the recommender from any other script on the network.

```python
import requests
#configure url, default is localhost
apiurl = 'http://127.0.0.1:8000/{}/{}/{}'
example_node = '0'
k = 10
r = requests.get(apiurl.format('knn',example_node,k))
r.json()
{0: {'neighbors': [0, 114546, 118173, 123258, 174705, 99438, 51354, 119874, 203176, 101864], 'distances': [0.9999998807907104, 0.9962959289550781, 0.9962303042411804, 0.9961680173873901, 0.9961460828781128, 0.9961054921150208, 0.9961045980453491, 0.9960995316505432, 0.9960215091705322, 0.9960126280784607]}}
```

Because we use a trained faiss index for our deployed API backend, requests should be returned very quickly even for large graphs. For the Reddit post recommender described above, the default API responds in about 82ms.

```python
import random
%timeit r = requests.get(apiurl.format('knn',random.randint(0,232964),k))
82.3 ms ± 5.42 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

## Save and Load

If you are creating a very large graph (millions of nodes and edges), you will want to save your created graph and model weights to disk, so that you will not have to process the raw edge data or train the embeddings again. You can save and load all of the necessary information to restore your GraphRecommeder in a single line. 

```python
sage.save('/example/directory')
```
You can likewise restore your session in a single line. 

```python
sage = GraphRecommender.load('/example/directory')
```

Note that the loading method is a classmethod, so you do not need to initialize a new instance of GraphRecommeder to restore from disk. The save and load functionality keeps track of the args you used to initialize the class for you.
