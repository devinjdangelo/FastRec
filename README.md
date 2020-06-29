# GraphSimEmbed

Graph neural networks are capable of capturing the structure and relationships between nodes in a graph as dense vectors. 
With these dense vectors, we can identify pairs of nodes that are similar, identify communities and clusters, or train
a linear classification model with the dense vectors as inputs. 

This project automates the entire pipeline from node/edge graph data to generate embeddings, train and fine tune those embeddings, create and train a Facebook AI Similarity Search Index (faiss), and deploy an API to query the index over the network. GraphSimEmbed handles all of the boilerplate code, handling gpu/cpu memory management, and passing data between pytorch, Deep Graph Library (DGL), faiss, and fastapi. 

The code is intended to be as scalable as possible, with the only limitation being the memory available to store the graph. The code adapts the implementation of [GraphSage](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) from the [DGL reference implementation](https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage). GraphSimEmbed has been tested on graphs with up to 10 million nodes and 100 million edges and was able to generate and train embeddings, train a faiss index, and begin answering api queries in minutes. With sufficient memory, it should be able to scale to billions of nodes and edges. Distributed training is not currently implemented, but could further improve scalability. 

## Installation

A dockerfile is included with all dependencies needed. Simply clone the repo, build the dockerfile, and run the code in the built image. Nvidia docker is needed for gpu support. There is not currently a pip or conda package.

## Usage

As an example, we can generate embeddings for [Zachary's karate club](https://en.wikipedia.org/wiki/Zachary%27s_karate_club) graph. First, convert the graph into a node and edgelist format.

```python
edges = nx.to_pandas_edgelist(nx.karate_club_graph())
nodes = pd.read_csv('./karate_attributes.csv')
```

Then we can initialize an embedder, add the data, and generate node embeddings.

```python
from sse import SimilarityEmbedder
sage = SimilarityEmbedder(2,distance='l2')
sage.add_data(edges,nodes,nodeid='node',classid='community')
untrained_embeddings =  sage.embeddings
```
How do the embeddings look? Even with no training of the graph neural network weights, the embeddings don't do a terrible job  dividing the two communities. 

![untrained_example_supervised](https://github.com/devinjdangelo/GraphSimEmbed/blob/master/examples/graphics/untrained_example_supervised.png)

With one command, we can improve the embeddings with supervised learning with a triplet loss. 

```python
epochs, batch_size = 150, 15
sage.train(epochs, batch_size)
```
![supervisedgif](https://github.com/devinjdangelo/GraphSimEmbed/blob/master/examples/graphics/supervised.gif)
![150epochs_supervised_trained](https://github.com/devinjdangelo/GraphSimEmbed/blob/master/examples/graphics/150epochs_supervised_trained.png)

The trained embeddings much more neatly divide the communities. But what about the more realistic scenario where we did not know the labels of all of the nodes in advance? We can instead train the embeddings in a fully unsupervised manner. 

![unsupervisedgif](https://github.com/devinjdangelo/GraphSimEmbed)
![150epochs_unsupervised_trained](https://github.com/devinjdangelo/GraphSimEmbed)

What if we have a very large graph which is expensive and slow to train? Often, the untrained performance of the embeddings will improve if we increate the size of our graph neural network (in terms of width and # of parameters).  

```python
sage = SimilarityEmbedder(2,distance='l2',feature_dim=512,hidden_dim=512)
untrained_embeddings_large =  sage.embeddings
```
![untrained_example_large](https://github.com/devinjdangelo/GraphSimEmbed)

This looks nearly as good as the trained version of the small network, but no training was required! Once we have embeddings that we are happy with, we can query a specific node or nodes to get its nearest neighbors in a single line.

```python
#what are the 5 nearest neighbors of node 0, the Admin, and 33, the Instructor?
sage.query_neighbors(['0','33'],k=5)
{'0': {'neighbors': ['0', '13', '16', '6', '5'], 'distances': [0.0, 0.001904212054796517, 0.005100540816783905, 0.007833012379705906, 0.008420777507126331]}, '33': {'neighbors': ['33', '27', '31', '28', '32'], 'distances': [0.0, 0.0005751167191192508, 0.0009900123113766313, 0.001961079193279147, 0.006331112235784531]}}
```
Each nodes nearest neighbor is itself with a distance of 0. The Admin is closest to nodes 13, 16, 6, and 5, all of which are in fact part of the Admin community. The Instructor is closest to 27, 31, 28, and 32, all of which are part of the Instructor community. 

## Recommender API

We can share the reccomender system as an API in a single line. No args are needed to test over localhost, but we can optionally pass in any args accepted by [uvicorn](https://www.uvicorn.org/deployment/).

```python
host, port = 127.0.0.1, 8000
sage.start_api(host=host,port=port)
```
Now we can query the reccomender from any other script on the network.

```python
import requests
#configure url, default is localhost
apiurl = 'http://127.0.0.1:8000/{}/{}/{}'
example_node = '0'
k = 5
r = requests.get(apiurl.format('knn',nodeid,k))
r.json()
{'0': {'neighbors': ['0', '13', '16', '6', '5'], 'distances': [0.0, 0.001904212054796517, 0.005100540816783905, 0.007833012379705906, 0.008420777507126331]}}
```

This method of starting the API is conveinient but has some downsides in the current implementation. Some data will be duplicated in memory, so if your graph is taking up most of your current memory this deployment may fail. You can avoid this issue by instead launching the API from a separate script using uvicorn directly.

```bash
uvicorn SageAPI:app
```

## Save and Load

If you are creating a very large graph (millions of nodes and edges), you will want to save your created graph and model weights to disk, so that you will not have to process the raw edge data or train the embeddings again. You can save and load all of the necessary information to restore your SimilarityEmbedder in a single line. 

```python
sage.save('/example/directory')
```
You can likewise restore your session in a single line. 

```python
sage = SimilarityEmbedder.load('/example/directory')
```

Note that the loading method is a classmethod, so you do not need to initialize a new instance of SimilarityEmbedder to restore from disk. The save and load functionality keeps track of the args you used to initialize the class for you.