# GraphSimEmbed

For graphs with many classes (thousands or millions) of node and only a few examples per class, 
cross entropy classification loss is impractical. Instead, we can output a fixed dimension embedding and
classify or cluster nodes by the pairwise distance between them in the embedding space. 

## Files

* GCNSimEmbed.py uses a graph convolutional network trained with triplet loss based on either MSE or cosine distance. 
Cannot scale since it requires the full graph adjacency matrix for each training step.
* SageSimEmbed.py uses GraphSage stochastic minibatch training method to scale to arbitrarily large graphs. Adapted from [DGL reference implementation](https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage)  
* geosim.py simulates graph data with arbitrary number of classes and a random number of examples per class between 1-3.

To replicate the below experiments:

* docker build -t dgl .
* ./dockerrun.sh
* ./run_experiments.sh

## Results
<img src="https://github.com/devinjdangelo/GraphSimEmbed/blob/master/Results/topn20_20_2_mse_1.0.png" alt="drawing" width="500"/>
<img src="https://github.com/devinjdangelo/GraphSimEmbed/blob/master/Results/training_20_20_2_mse_1.0.gif" alt="drawing" width="500"/>
