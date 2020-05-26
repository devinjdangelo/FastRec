# GraphSimEmbed

For graphs with many classes (thousands or millions) of node and only a few examples per class, 
cross entropy classification loss is impractical. Instead, we can output a fixed dimension embedding and
classify or cluster nodes by the pairwise distance between them in the embedding space. 

## Files

* GCNSimEmbed.py uses a graph convolutional network trained with triplet loss based on either MSE or cosine distance. 
Cannot scale since it requires the full graph adjacency matrix for each training step.
* SageSimEmbed.py uses [GraphSage](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) stochastic minibatch training method to scale to arbitrarily large graphs. Adapted from [DGL reference implementation](https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage).  
* geosim.py simulates graph data with arbitrary number of classes and a random number of examples per class between 1-3.

To replicate the below experiments, clone the repo and cd into it. Then run:

* mkdir ./chunks
* mkdir ./ims
* docker build -t dgl .
* ./dockerrun.sh
* ./run_experiments.sh

## Results

First experiment has a small number of classes in a 2d embedding space for visualization purposes. The classes of all nodes are known at training time, so this is fully supervised.

<img src="https://github.com/devinjdangelo/GraphSimEmbed/blob/master/Results/topn20_20_2_mse_1.0.png" alt="drawing" width="500"/>

The below visualizes all nodes in the 2-d embedding space for each epoch in trianing. The numbers next to each dot represent the true class label. The dots of the same class gradually cluster together.

<img src="https://github.com/devinjdangelo/GraphSimEmbed/blob/master/Results/training_20_20_2_mse_1.0.gif" alt="drawing" width="500"/>

Training is much faster and more stable (albeit slower on a per epoch basis) using GCN aggregator rather than the mean aggregator. 

<img src="https://github.com/devinjdangelo/GraphSimEmbed/blob/master/Results/topn20_25_2_mse_1.0_1.png" alt="drawing" width="500"/>

<img src="https://github.com/devinjdangelo/GraphSimEmbed/blob/master/Results/topn20_25_2_mse_1.0_1.gif" alt="drawing" width="500"/>

Next, test with 10k classes (~180k nodes and 1.3m edges). This graph is too large to fit in the memory of a single gpu without minibatch training. Still, all node classes are known at training time so this is fully supervised.

<img src="https://github.com/devinjdangelo/GraphSimEmbed/blob/master/Results/topn10000_1000_32_cosine_1.png" alt="drawing" width="500"/>

Finally, repeat with only 50% of node classes known at training time. This is now semi-supervised training. The accuracy does not improve beyond 50%. 

<img src="https://github.com/devinjdangelo/GraphSimEmbed/blob/master/Results/topn10000_1000_32_cosine_0.5.png" alt="drawing" width="500"/>

Using the GCN aggregator rather than the mean aggregator results in much greater performance of the untrained model. Fine tuning with unsupervised loss and a small number of known labels (10%) results in near 100% accuracy.

<img src="https://github.com/devinjdangelo/GraphSimEmbed/blob/master/Results/topn10000_1000_16_cosine_0.1_0.5.png" alt="drawing" width="500"/>
