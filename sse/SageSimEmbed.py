# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:39:20 2020

@author: djdev
"""

import pandas as pd
import networkx as nx
import time
import numpy as np
import pathlib
from math import ceil
import argparse
import itertools as it
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import tqdm
import imageio
import pickle

import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import dgl.nn.pytorch as dglnn

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import faiss

from .torchmodels import *

  
class SimilarityEmbedder:
    def __init__(self, embedding_dim,
                        feature_dim = None,
                        hidden_dim = None,
                        hidden_layers = 2,
                        dropout = 0,
                        agg_type = 'gcn',
                        distance = 'cosine',
                        torch_device = 'cpu',
                        faiss_gpu = False,
                        inference_batch_size = 10000,
                        p_train = 1):
        """Generates embeddings for graph data such that embeddings close by a given distance metric are
        'similar'. Embeddings can be used to predict which nodes belong to the same class. The embeddings can be
        trained with triplet loss in a fully supervised, semi-supervised or fully unsupervised manner. GraphSage
        is used to allow minibatch training. Uses faiss index to allow extremely fast query times for most similar
        nodes to a query node even for graphs with billions of nodes. Memory is likely to be the limiting factor before
        query times. 

        Args
        ----
        embedding_dim : the dimension of the final output embedding used for similarity search
        feature_dim : the dimension of the input node features, currently only allowed to be 
                        a trainable embedding. In the future should allow external node features.
                        defaults to 2*hidden_dim
        hidden_dim : the dimension of the intermediate hidden layers, defaults to 2*embedding dim.
        hidden_layers : number of hidden layers. Embeddings can collpase to a single value if this 
                        is set too high. Defaults to 2.
        dropout : whether to apply a dropout layer after hidden layers of GraphSAge. Defaults to 0,
                    which means there is no Dropout applied.
        agg_type : aggregation function to apply to GraphSage. Valid options are 'mean', 'lstm', and 'gcn'
                   aggregation. See GraphSage paper for implementation details. Defaults to gcn which performs
                   well for untrained networks.
        distance : distance metric to use for similarity search. Valid options are mse and cosine. Defaults to cosine.
        torch_device : computation device to place pytorch tensors on. Valid options are any valid pytorch device. Defaults 
                to cpu.
        faiss_gpu : whether to use gpu to accelerate faiss searching. Note that it will compete with pytorch for gpu memory.
        inference_batch_size : number of nodes to compute per batch when computing all embeddings with self.net.inference.
                                defaults to 10000 which should comfortably fit on most gpus and be reasonably efficient on cpu.
        p_train : the proportion of nodes with known class labels to use for training defaults to 1 
        """
        self.embedding_dim = embedding_dim
        self.device = torch_device 
        self.inference_batch_size = inference_batch_size
        assert p_train<=1 and p_train>=0
        self.p_train = p_train
        self.faiss_gpu = faiss_gpu

        self.distance_metric = distance
        if self.distance_metric == 'cosine':
            self.distance_function = lambda t1,t2 : F.cosine_embedding_loss(t1,
                                                t2,
                                                th.ones(t1.shape[0]).to(self.device),reduce=False)
        elif self.distance_metric == 'mse':
            self.distance_function = lambda t1,t2 : th.sum(F.mse_loss(t1,t2,reduce=False),dim=1)
        else:
            raise ValueError('distance {} is not implemented'.format(self.distance))

        hidden_dim = embedding_dim*2 if hidden_dim is None else hidden_dim
        feature_dim = hidden_dim*2 if feature_dim is None else feature_dim
        self.feature_dim = feature_dim
        self.net = SAGE(feature_dim, hidden_dim, embedding_dim, hidden_layers, F.relu, dropout, agg_type)
        self.net.to(self.device)

        self._embeddings = None 
        self._index = None 
        self._masks_set = False

        self.node_ids = pd.DataFrame(columns=['id','intID','classid'])
        self.G = DGLGraph()
        self.G.readonly(True)
        self.G = dgl.as_heterograph(self.G)


    def add_data(self, edgelist, nodeid1, nodeid2, classid1=None, classid2=None):
        """Updates graph data with a new edgelist. Maps each nodeid to an integer id. 
        input nodeids can be any unique identifier (strings, floats, integers ect...).
        They will internally be mapped to a sequential integer id which DGL uses. self.nodeids
        keeps track of the mappings between input identifiers and the sequential integer ids.
        If you run out of memory when calling this method, split edglist into chunks and call
        this method once for each chunk.

        Args
        ----
        edgelist : dataframe with edges between nodes, edges are assumed to be bidrectional and 
                    should only be in the dataframe once (e.g. a <-> b means do not include b <-> a)
        nodeid1 : column name in edgelist which uniquely identifies node1
        nodeid2 : columns name in edgelist which uniquely identifies node2
        classid1 : optional column name in edgelist which assigns a class to node1
        classid2 : optional column name in edgelist which assigns a class to node2
        p_train : proportion of newly added nodes which have an assigned class to use in training set
        p_test : proportion of newly added nodes which have an assigned class to use in testing set"""

        if classid1 is not None or classid2 is not None:
            assert classid1 is not None and classid2 is not None
        else:
            classid1 = 'classid1'
            classid2 = 'classid2'
            edgelist[classid1] = None 
            edgelist[classid2] = None

        edgelist = edgelist[[nodeid1,nodeid2,classid1,classid2]]
        
        n1 = edgelist[[nodeid1,classid1]].drop_duplicates()
        n1.columns = ['id', 'classid']
        n2 = edgelist[[nodeid2,classid2]].drop_duplicates()
        n2.columns = ['id', 'classid']

        nodes = pd.concat([n1,n2])

        nodes = nodes.merge(self.node_ids,on='id',how='left',suffixes=('','_merged'))
        num_new_nodes = int(nodes.intID.isna().sum())
        current_maximum_id = self.node_ids.intID.max()
        start = (current_maximum_id+1)
        if np.isnan(start):
            start = 0
        end = start + num_new_nodes
        new_nodes = pd.isna(nodes.intID)
        nodes.loc[new_nodes,'intID'] = list(range(start,end))
        update_nodes = nodes[new_nodes]
        update_nodes = update_nodes[['id','intID','classid']]
        self.node_ids = pd.concat([self.node_ids,update_nodes])

        edgelist = edgelist.merge(nodes,left_on=nodeid1,right_on='id',how='left')
        edgelist = edgelist.merge(nodes,left_on=nodeid2,right_on='id',how='left',suffixes=('','2'))
        edgelist = edgelist[['intID','intID2']]

        self.G = dgl.as_immutable_graph(self.G)
        self.G.readonly(False)
        self.G.add_nodes(num_new_nodes)
        self.G.add_edges(edgelist.intID.tolist(),edgelist.intID2.tolist())
        self.G.add_edges(edgelist.intID2.tolist(),edgelist.intID.tolist())
        self.G.readonly()
        self.G = dgl.as_heterograph(self.G)

        #reset internal flag that embeddings and index need to be recomputed
        self._embeddings = None 
        self._index = None 
        self._masks_set = False

    def set_masks(self):
        """Sets train, test, and relevance masks. Needs to be called once after data as been added to graph.
        self.train and self.evaluate automatically check if this needs to be called and will call it, but
        it can also be called manually. Can be called a second time manually to reroll the random generation
        of the train and test sets."""

        self.node_ids = self.node_ids.sort_values('intID')
        self.labels = self.node_ids.classid.to_numpy()

        #is relevant mask indicates the nodes which we know the class of
        self.is_relevant_mask = pd.isna(self.node_ids.classid).to_numpy()

        self.train_mask =  np.random.choice(
        a=[False,True],size=(len(self.node_ids)),p=[1-self.p_train,self.p_train])

        #test set is all nodes other than the train set unless train set is all
        #nodes and then test set is the same as train set.
        if self.p_train != 1:
            self.test_mask = np.logical_not(self.train_mask)
        else:
            self.test_mask = self.train_mask

        #do not include any node without a classid in either set
        self.train_mask = np.logical_and(self.train_mask,self.is_relevant_mask)
        self.test_mask = np.logical_and(self.test_mask,self.is_relevant_mask)

        self.embed = nn.Embedding(len(self.node_ids),self.feature_dim)
        self.G.ndata['features'] = self.embed.weight
        self.features = self.embed.weight
        self.features.to(self.device)
        self.embed.to(self.device)

        self._masks_set = True

    @property
    def embeddings(self):
        """Updates all node embeddings if needed and returns the embeddings.
        Simple implementation of a cached property.

        Returns
        -------
        embeddings node x embedding_dim tensor"""

        if self._embeddings is None:
            print('computing embeddings for all nodes...')
            with th.no_grad():
                self._embeddings = self.net.inference(self.G, self.features,self.inference_batch_size,self.device).detach().cpu().numpy()
        return self._embeddings

    @property
    def index(self):
        """Creates a faiss index for similarity searches over the node embeddings.
        Simple implementation of a cached property.

        Args
        ----
        embeddings : the embeddings to add to faiss index
        use_gpu : whethern to store the index on gpu and use gpu for compute

        Returns
        -------
        a faiss index of input embeddings"""

        if self._index is None:
            if self.distance_metric=='cosine':
                self._index  = faiss.IndexFlatIP(self.embedding_dim)
                normalized_embeddings = np.copy(self.embeddings)
                #this function operates in place so np.copy any views into a new array before using.
                faiss.normalize_L2(normalized_embeddings)
                embeddings = normalized_embeddings
            elif self.distance_metric=='mse':
                self._index = faiss.IndexFlatL2(self.embedding_dim)
                embeddings = self.embeddings
            if self.faiss_gpu:
                GPU = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(GPU, 0, self._index)

            self._index.add(embeddings)

        return self._index

    def _search_index(self,inputs,k):
        """Directly searches the faiss index and 
        returns the k nearest neighbors of inputs

        Args
        ----
        inputs : the vectors to search against the faiss index
        k : how many neighbors to lookup
        label_nodes : lookup labels of nodes or just return integer ids

        Returns
        -------
        D, I distance numpy array and neighbors array from faiss"""
        if self.distance_metric == 'cosine':
            inputs = np.copy(inputs)
            faiss.normalize_L2(inputs)
        D, I = self.index.search(inputs,k)
        return D,I

    def query_neighbors(self, nodelist, k):
        """For each query node in nodelist, return the k closest neighbors in the 
        embedding space.

        Args
        ----
        nodelist : list of node identifiers to query
        k : number of neighbors to return

        Returns
        -------
        dictionary of neighbors for each querynode and corresponding distance"""

        intids = [self.node_ids.loc[self.node_ids.id == node].intID.iloc[0]
                    for node in nodelist]
        inputs = self.embeddings[intids,:]
        D, I = self._search_index(inputs,k)
        I = [[self.node_ids.loc[self.node_ids.intID==neighbor].id.iloc[0] for neighbor in neighbors] for neighbors in I]
        output = {node:{'neighbors':i,'distances':d} for node, d, i in zip(nodelist,D,I)}
        return output

    def evaluate(self, test_only=False):
        """Evaluates performance of current embeddings

        Args
        ----
        test_only : whether to only test the performance on the test set. If 
                    false, all nodes with known class will be tested.

        Returns
        -------
        P at least 1 correct neighbors are in top5, and top1 respectively"""
        self.net.eval()

        if not self._masks_set:
            self.set_masks()

        mask = self.test_mask if test_only else self.is_relevant_mask
        test_labels = self.labels[mask]

        test_embeddings = self.embeddings[self.test_mask]
        _, I = self._search_index(test_embeddings,6)

        ft1, ft5 = [], []
        for node, neighbors in enumerate(I):
            label = test_labels[node]
            neighbor_labels = [self.labels[n] for n in neighbors[1:]]
            ft1.append(label==neighbor_labels[0])
            ft5.append(label in neighbor_labels)

        return np.mean(ft5), np.mean(ft1)

    @staticmethod
    def setup_pairwise_loss_tensors(labelsnp):
        """Accepts a list of labels and sets up indexers which can be used
        in a triplet loss function along with whether each pair is a positive or
        negative example.

        Args
        ----
        labelsnp : numpy array of labels

        Returns
        -------
        idx1 : indexer array for left side comparison
        idx2 : indexer array for right side comparison
        target : array indicating whether left and right side are the same or different"""

        idx1 = []
        idx2 = []
        target = []
        for i,l in enumerate(labelsnp):
            ids = list(range(len(labelsnp)))
            for j,other in zip(ids[i+1:],labelsnp[i+1:]):
                if other==l:
                    idx1.append(i)
                    idx2.append(j)
                    target.append(1)
                else:
                    idx1.append(i)
                    idx2.append(j)
                    target.append(-1)

        return idx1, idx2, target

    def triplet_loss(self,embeddings,labels):
        """For a given tensor of embeddings and corresponding labels, 
        returns a triplet loss maximizing distance between negative examples
        and minimizing distance between positive examples

        Args
        ----
        embeddings : pytorch tensor of embeddings to be trained
        labels : labels indicating which embeddings are equivalent"""
        
        batch_relevant_nodes = [i for i,l in enumerate(labels) if pd.isna(l)]
        embeddings = embeddings[batch_relevant_nodes]
        labels = labels[batch_relevant_nodes]
        idx1,idx2,target = self.setup_pairwise_loss_tensors(labels)

        losstarget = th.tensor(target).to(self.device)

        if self.distance_metric=='cosine':
            input1 = embeddings[idx1]
            input2 = embeddings[idx2]
            loss = F.cosine_embedding_loss(input1,
                                            input2,
                                            losstarget,
                                            margin=0.5)
        elif self.distance_metric=='mse':
            idx1_pos = [idx for i,idx in enumerate(idx1) if target[i]==1]
            idx1_neg = [idx for i,idx in enumerate(idx1) if target[i]==-1]

            idx2_pos = [idx for i,idx in enumerate(idx2) if target[i]==1]
            idx2_neg = [idx for i,idx in enumerate(idx2) if target[i]==-1]

            input1_pos = embeddings[idx1_pos]
            input2_pos = embeddings[idx2_pos]

            input1_neg = embeddings[idx1_neg]
            input2_neg = embeddings[idx2_neg]

            loss_pos = F.mse_loss(input1_pos,input2_pos)
            loss_neg = th.mean(th.max(th.zeros(input1_neg.shape[0]).to(self.device),0.25-th.sum(F.mse_loss(input1_neg,input2_neg,reduce=False),dim=1)))

            loss = loss_pos + loss_neg

        else:
            raise ValueError('distance {} is not implemented'.format(self.distance))

        return loss 
       

    def train(self,epochs,
                    batch_size,
                    test_every_n_epochs = 1,
                    unsupervised = False,
                    learning_rate = 1e-2,
                    fanouts = [10,25],
                    neg_samples = 1):
        """Trains the network weights to improve the embeddings. Can train via supervised learning with triplet loss,
        semisupervised learning with triplet loss, or fully unsupervised learning.

        Args
        ----
        epochs : number of training passes over the data
        batch_size : number of seed nodes for building the training graph
        test_every_n_epochs : how often to do a full evaluation of the embeddings, expensive for large graphs
        unsupervised : whether to train completely unsupervised
        learning_rate : learning rate to use in the adam optimizer
        fanouts : number of neighbors to sample at each layer for GraphSage
        neg_samles : number of negative samples to use in unsupervised loss"""

        if not self._masks_set:
            self.set_masks()

        optimizer = th.optim.Adam(it.chain(self.net.parameters(),self.embed.parameters()), lr=learning_rate)

        if not unsupervised:
            sampler = NeighborSampler(self.G, [int(fanout) for fanout in fanouts])
            data = list(range(len(self.node_ids)))
        else:
            sampler = UnsupervisedNeighborSampler(self.G, [int(fanout) for fanout in fanouts],neg_samples)
            data = np.nonzero(self.train_mask)[0]
            unsup_loss_fn = CrossEntropyLoss()
            unsup_loss_fn.to(self.device)

        dataloader = DataLoader(
                            dataset=data,
                            batch_size=batch_size,
                            collate_fn=sampler.sample_blocks,
                            shuffle=True,
                            drop_last=True,
                            num_workers=1)

        

        
        testtop5,testtop1 = self.evaluate(test_only=True)

        print("Test Top5 {:.4f} | Test Top1 {:.4f}".format(
                testtop5,testtop1))

        loss_history = []
        top5_history = [testtop5]
        top1_history = [testtop1]

        for epoch in range(1,epochs+1):
            
            for step,data in enumerate(dataloader):
                #sup_blocks, unsupervised_data = data 
                #pos_graph, neg_graph, unsup_blocks = unsupervised_data


                self.net.train()

                # these names are confusing because "seeds" are the input
                # to neighbor generation but the output in the sense that we 
                # output their embeddings based on their neighbors...
                # the neighbors are the inputs in the sense that they are what we
                # use to generate the embedding for the seeds.
                if not unsupervised:
                    sup_blocks = data
                    sup_input_nodes = sup_blocks[0].srcdata[dgl.NID]
                    sup_seeds = sup_blocks[-1].dstdata[dgl.NID]

                    sup_batch_inputs = self.G.ndata['features'][sup_input_nodes].to(self.device)
                    sup_batch_labels = self.labels[sup_seeds]

                    sup_embeddings = self.net(sup_blocks, sup_batch_inputs)

                    loss = self.triplet_loss(sup_embeddings,sup_batch_labels)
                else:
                    pos_graph, neg_graph, unsup_blocks = data
                    unsup_input_nodes = unsup_blocks[0].srcdata[dgl.NID]
                    unsup_seeds = unsup_blocks[-1].dstdata[dgl.NID]

                    unsup_batch_inputs = self.G.ndata['features'][unsup_input_nodes].to(self.device)

                    unsup_embeddings =self.net(unsup_blocks,unsup_batch_inputs)
                    loss = unsup_loss_fn(unsup_embeddings, pos_graph, neg_graph)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #once the parameters change we no longer know the new embeddings for all nodes
                self._embeddings = None 

                print("Epoch {:05d} | Step {:0.1f} | Loss {:.8f} | Mem+Maxmem {:.3f} / {:.3f}".format(
                        epoch, step, loss.item(), th.cuda.memory_allocated()/(1024**3),th.cuda.max_memory_allocated()/(1024**3)))
            
            loss_history.append(loss.item())
            if epoch % test_every_n_epochs == 0 or epoch==epochs:
                testtop5,testtop1 = self.evaluate(test_only=True)
                top5_history.append(testtop5)
                top1_history.append(testtop1)

                print("Epoch {:05d} | Loss {:.8f} | Test Top5 {:.4f} | Test Top1 {:.4f}".format(
                        epoch, loss.item(),testtop5,testtop1))                

    def save(self, filepath):
        """Save embeddings, model weights, and graph data to disk so it can be restored later

        Args
        ----
        filepath : str path on disk to save files"""
        pass 

    def load(self, filepath):
        """restore embeddings, model weights and graph data from disk.

        Args
        ----
        filepath : str path on disk to load from"""




