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
import tqdm
import os

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
import uvicorn

from torchmodels import *

#this is the maximum edges we will ad at once to keep temp memory usage down
MAX_ADD_EDGES = 1e6

# this is the target ratio of nodes to faiss clusters for index training
# roughly matches what the faiss warning messages suggest in testing
FAISS_NODES_TO_CLUSTERS = 1000

#Arbitrary... not sure what this should be long term. Depends on memory usage
#which I haven't tested thoroughly yet.
MAXIMUM_FAISS_CLUSTERS = 10000
  
class GraphRecommender:
    """Rapidly trains similarity embeddings for graphs and generates recomendations

    Attributes
    ----------
    G : DGL Graph object
        Current DGL graph for all added data with self.add_data
    node_ids : pandas data frame
        Contains mapping from user provided nodeids to DGL and faiss compatable integer ids.
        Also contains various flags which identify properties and classes of the nodes.
    """

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
                        p_train = 1,
                        train_faiss_index = False):
        """Generates embeddings for graph data such that embeddings close by a given distance metric are
        'similar'. Embeddings can be used to predict which nodes belong to the same class. The embeddings can be
        trained with triplet loss in a fully supervised, semi-supervised or fully unsupervised manner. GraphSage
        is used to allow minibatch training. Uses faiss index to allow extremely fast query times for most similar
        nodes to a query node even for graphs with billions of nodes. Memory is likely to be the limiting factor before
        query times. 

        Args
        ----
        embedding_dim : int
            the dimension of the final output embedding used for similarity search
        feature_dim : int
            the dimension of the input node features, currently only allowed to be 
            a trainable embedding. In the future should allow external node features.
            defaults to 2*hidden_dim
        hidden_dim : int
            the dimension of the intermediate hidden layers, defaults to 2*embedding dim.
        hidden_layers : int
            number of hidden layers. Embeddings can collpase to a single value if this 
            is set too high. Defaults to 2.
        dropout : float
            whether to apply a dropout layer after hidden layers of GraphSAge. Defaults to 0,
            which means there is no Dropout applied.
        agg_type : str
            aggregation function to apply to GraphSage. Valid options are 'mean', 'lstm', and 'gcn'
            aggregation. See GraphSage paper for implementation details. Defaults to gcn which performs
            well for untrained networks.
        distance : str
            distance metric to use for similarity search. Valid options are l2 and cosine. Defaults to cosine.
        torch_device : str
            computation device to place pytorch tensors on. Valid options are any valid pytorch device. Defaults 
            to cpu.
        faiss_gpu : bool
            whether to use gpu to accelerate faiss searching. Note that it will compete with pytorch for gpu memory.
            inference_batch_size : number of nodes to compute per batch when computing all embeddings with self.net.inference.
            defaults to 10000 which should comfortably fit on most gpus and be reasonably efficient on cpu.
        p_train : float
            the proportion of nodes with known class labels to use for training defaults to 1 
        train_faiss_index : bool
            whether to train faiss index for faster searches. Not reccomended for training since brute force
            will actually be faster than retraining the index at each test iteration. Can be used for api to speed
            up response times.
        """
        self.embedding_dim = embedding_dim
        self.device = torch_device 
        self.inference_batch_size = inference_batch_size
        assert p_train<=1 and p_train>=0
        self.p_train = p_train
        self.faiss_gpu = faiss_gpu
        self.train_faiss = train_faiss_index

        self.distance_metric = distance
        if self.distance_metric == 'cosine':
            self.distance_function = lambda t1,t2 : F.cosine_embedding_loss(t1,
                                                t2,
                                                th.ones(t1.shape[0]).to(self.device),reduce=False)
        elif self.distance_metric == 'l2':
            self.distance_function = lambda t1,t2 : th.sum(F.mse_loss(t1,t2,reduce=False),dim=1)
        else:
            raise ValueError('distance {} is not implemented'.format(self.distance))

        hidden_dim = embedding_dim*4 if hidden_dim is None else hidden_dim
        feature_dim = hidden_dim*2 if feature_dim is None else feature_dim
        self.feature_dim = feature_dim
        self.net = SAGE(feature_dim, hidden_dim, embedding_dim, hidden_layers, F.relu, dropout, agg_type)
        self.net.to(self.device)

        self._embeddings = None 
        self._index = None 
        self._masks_set = False

        self.node_ids = pd.DataFrame(columns=['id','intID','classid','feature_flag'])
        self.G = DGLGraph()

        #hold init args in memory in case needed to save to disk for restoring later
        self.initargs = (embedding_dim,
                        feature_dim,
                        hidden_dim,
                        hidden_layers,
                        dropout,
                        agg_type,
                        distance,
                        torch_device,
                        faiss_gpu,
                        inference_batch_size,
                        p_train,
                        train_faiss_index)


    def add_nodes(self, nodearray, skip_duplicates=False):
        """Define nodes by passing an array (or array like object). Nodes
        can be identified by any data type (even mixed data types), but each
        node must be unique. An exception is raised if all nodes are not unique
        including if the same node is attempted to be added in two calls to this 
        method. Each node is mapped to a unique integer id based on the order
        they are added.

        Args
        ----
        nodearray : numpy array (or array-like object)
            array containing the identifiers of each node to be added
        skip_duplicates : bool
            if true, ignore nodes which have already been added. If False, raise error.
        """
        
        ninputnodes = len(nodearray)
        nodedf = pd.DataFrame(nodearray, columns=['id'])

        if len(nodedf) != len(nodedf.drop_duplicates()):
            raise ValueError('Provided nodeids are not unique. Please pass an array of unique identifiers.')

        nodes_already_exist = nodedf.merge(self.node_ids,on='id',how='inner')
        if len(nodes_already_exist)>0 and not skip_duplicates:
            raise ValueError(
            'Some provided nodes have already been added to the graph. See node_ids.ids.')
        elif len(nodes_already_exist)>0 and skip_duplicates:
            #get rid of the duplicates
            nodes_already_exist['dropflag'] = True 
            nodedf = nodedf.merge(nodes_already_exist,on='id',how='left')
            nodedf['dropflag'] = ~pd.isna(nodedf.dropflag)
            nodedf = nodedf.drop(nodedf[nodedf.dropflag].index)
            nodedf = nodedf[['id']]
            

        current_maximum_id = self.node_ids.intID.max()
        num_new_nodes = len(nodedf)

        start = (current_maximum_id+1)
        if np.isnan(start):
            start = 0
        end = start + num_new_nodes

        nodedf['intID'] = range(start,end)
        nodedf['classid'] = None 
        nodedf['feature_flag'] = False

        self.node_ids = pd.concat([self.node_ids,nodedf])

        self._masks_set = False

        if self.G.is_readonly:
            self.G = dgl.as_immutable_graph(self.G)
            self.G.readonly(False)
        self.G.add_nodes(num_new_nodes)

        self._masks_set = False
        self._embeddings = None 
        self._index = None       


    def add_edges(self, n1, n2):
        """Adds edges to the DGL graph. Nodes must be previously defined by
        add_nodes or an exception is raised. Edges are directed. To define
        a undirected graph, include both n1->n2 and n2->n1 in the graph.

        Args
        ----
        n1 : numpy array (or array-like object)
            first node in the edge (n1->n2)
        n2 : numpy array (or array-like object)
            second node in the edge (n1->n2)
        """
        edgedf_all = pd.DataFrame(n1,columns=['n1'])
        edgedf_all['n2'] = n2

        chunks = int(max(len(edgedf_all)//MAX_ADD_EDGES,1))
        edgedf_all = np.array_split(edgedf_all, chunks)

        if chunks>1:
            pbar = tqdm.tqdm(total=chunks)

        for i in range(chunks):
            edgedf = edgedf_all.pop()
            edgedf = edgedf.merge(self.node_ids,left_on='n1',right_on='id',how='left')
            edgedf = edgedf.merge(self.node_ids,left_on='n2',right_on='id',how='left',suffixes=('','2'))
            edgedf = edgedf[['intID','intID2']]

            if len(edgedf) != len(edgedf.dropna()):
                raise ValueError('Some edges do not correspond to any known node. Please add with add_nodes method first.')

            if self.G.is_readonly:
                self.G = dgl.as_immutable_graph(self.G)
                self.G.readonly(False)

            self.G.add_edges(edgedf.intID,edgedf.intID2)

            if chunks>1:
                pbar.update(1)

        if chunks>1:
            pbar.close()

        self._masks_set = False
        self._embeddings = None 
        self._index = None     

    def _update_node_ids(self,datadf):
        """Overwrites existing information about nodes with new info
        contained in a dataframe. Temporarily sets id as the index to use
        built in pandas update method aligned on index.

        Args
        ----
        datadf : data frame
            has the same structure as self.node_ids
        """

        datadf.set_index('id',inplace=True,drop=True)
        self.node_ids.set_index('id',inplace=True,drop=True)
        self.node_ids.update(datadf, overwrite=True)
        self.node_ids.reset_index(inplace=True)

    def update_labels(self,labels):
        
        """Updates nodes by adding a label (or class). Existing class label
        is overridden if one already exists. Any node which does not have a 
        known class has a label of None. Any data type can be a valid class 
        label except for None which is reserved for unknown class. All nodes
        included in the update must be previously defined by add_nodes or
        an exception is raised.

        Args
        ----
        labels : dictionary or pandas series
            maps node ids to label, i.e. classid. If pandas series the index acts as the dictionary key."""

        labeldf = pd.DataFrame(labels.items(), columns=['id','classid'])
        labeldf = labeldf.merge(self.node_ids,on='id',how='left',suffixes=('','2'))

        if labeldf['intID'].isna().sum() > 0:
            raise ValueError('Some nodes in update do not exist in graph. Add them first with add_nodes.')

        labeldf = labeldf[['id','intID','classid','feature_flag']]
        self._update_node_ids(labeldf)

        self._masks_set = False
        self._embeddings = None 
        self._index = None     

    def update_feature_flag(self,flags):
        """Updates node by adding a feature flag. This can be True or False.
        If the feature flag is True, the node will not be included in any 
        recommender index. It will still be included in the graph to enrich
        the embeddings of the other nodes, but it will never be returned as
        a recommendation as a similar node. I.e. if True this node is a feature
        of other nodes only and not interesting as an entity of its own right.

        Args
        ----
        flags : dictionary or pandas series
            maps node ids to feature flag. If pandas series the index acts as the dictionary key."""

        featuredf = pd.DataFrame(flags.items(), columns=['id','feature_flag'])
        featuredf = featuredf.merge(self.node_ids,on='id',how='left',suffixes=('','2'))

        if featuredf['intID'].isna().sum() > 0:
            raise ValueError('Some nodes in update do not exist in graph. Add them first with add_nodes.')

        featuredf = featuredf[['id','intID','classid','feature_flag']]
        self._update_node_ids(featuredf)

        self._masks_set = False
        self._embeddings = None 
        self._index = None     

    def set_masks(self):
        """Sets train, test, and relevance masks. Needs to be called once after data as been added to graph.
        self.train and self.evaluate automatically check if this needs to be called and will call it, but
        it can also be called manually. Can be called a second time manually to reroll the random generation
        of the train and test sets."""

        self.node_ids = self.node_ids.sort_values('intID')
        self.labels = self.node_ids.classid.to_numpy()

        #is relevant mask indicates the nodes which we know the class of
        self.is_relevant_mask = np.logical_not(pd.isna(self.node_ids.classid).to_numpy())

        #entity_mask indicates the nodes which we want to include in the faiss index
        self.entity_mask = np.logical_not(self.node_ids.feature_flag.to_numpy().astype(np.bool))

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
        self.train_mask = np.logical_and(self.train_mask,self.entity_mask)
        self.test_mask = np.logical_and(self.test_mask,self.is_relevant_mask)
        self.test_mask = np.logical_and(self.test_mask,self.entity_mask)

        if not self.G.is_readonly:
            self.embed = nn.Embedding(len(self.node_ids),self.feature_dim)
            self.G.readonly()
            self.G = dgl.as_heterograph(self.G)
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
                self._embeddings = self.net.inference(
                    self.G, self.features,self.inference_batch_size,self.device).detach().cpu().numpy()
        return self._embeddings

    @property
    def index(self):
        """Creates a faiss index for similarity searches over the node embeddings.
        Simple implementation of a cached property.

        Returns
        -------
        a faiss index with input embeddings added and optionally trained"""

        if not self._masks_set:
            self.set_masks()

        if self._index is None:
            if self.distance_metric=='cosine':
                self._index  = faiss.IndexFlatIP(self.embedding_dim)
                embeddings = np.copy(self.embeddings[self.entity_mask])
                #this function operates in place so np.copy any views into a new array before using.
                faiss.normalize_L2(embeddings)
            elif self.distance_metric=='l2':
                self._index = faiss.IndexFlatL2(self.embedding_dim)
                embeddings = self.embeddings[self.entity_mask]
            
            if self.train_faiss:
                training_points = min(
                    len(self.node_ids)//FAISS_NODES_TO_CLUSTERS+1,
                    MAXIMUM_FAISS_CLUSTERS)
                self._index = faiss.IndexIVFFlat(self._index, self.embedding_dim, training_points)
                self._index.train(embeddings)

            self._index.add(embeddings)

            if self.faiss_gpu:
                GPU = faiss.StandardGpuResources()
                self._index = faiss.index_cpu_to_gpu(GPU, 0, self._index)


        return self._index

    def _search_index(self,inputs,k):
        """Directly searches the faiss index and 
        returns the k nearest neighbors of inputs

        Args
        ----
        inputs : numpy array np.float
            the vectors to search against the faiss index
        k : int
            how many neighbors to lookup

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
        nodelist : list of any types castable to stirng
            list of node identifiers to query
        k : int
            number of neighbors to return

        Returns
        -------
        dictionary of neighbors for each querynode and corresponding distance"""

        if not self._masks_set:
            self.set_masks()

        try:
            intids = [self.node_ids.loc[self.node_ids.id == node].intID.iloc[0]
                        for node in nodelist]
        except IndexError:
            intids = [self.node_ids.loc[self.node_ids.id == int(node)].intID.iloc[0]
                        for node in nodelist]

        inputs = self.embeddings[intids,:]
        D, I = self._search_index(inputs,k)
        faissid_to_nodeid = self.node_ids.id.to_numpy()[self.entity_mask]
        I = [[faissid_to_nodeid[neighbor] for neighbor in neighbors] for neighbors in I]
        output = {node:{'neighbors':i,'distances':d.tolist()} for node, d, i in zip(nodelist,D,I)}
        return output

    def evaluate(self, test_levels=[5,1], test_only=False):
        """Evaluates performance of current embeddings

        Args
        ----
        test_only : bool
            whether to only test the performance on the test set. If 
            false, all nodes with known class will be tested.
        test_levels : list of ints
            each entry is a number of nearest neighbors and we will test
            if at least one of the neighbors at each level contains a correct
            neighbor based on node labels. We also test the 
            total share of the neighbors that have a correct label.

        Returns
        -------
        dictionary containing details of the performance of the model at each level
        """

        self.net.eval()

        if not self._masks_set:
            self.set_masks()

        mask = self.test_mask if test_only else self.is_relevant_mask
        test_labels = self.labels[mask]
        faiss_labels = self.labels[self.entity_mask]

        test_embeddings = self.embeddings[mask]

        #we need to return the maximum number of neighbors that we want to test
        #plus 1 since the top neighbor of each node will always be itself, which
        #we exclude.
        _, I = self._search_index(test_embeddings,max(test_levels)+1)

        performance = {level:[] for level in test_levels}
        performance_share = {level:[] for level in test_levels}
        for node, neighbors in enumerate(I):
            label = test_labels[node]
            neighbor_labels = [faiss_labels[n] for n in neighbors[1:]]
            for level in test_levels:
                correct_labels = np.sum([label==nl for nl in neighbor_labels[:level]])
                #at least one label in the neighbors was correct
                performance[level].append(correct_labels>0)
                #share of labels in the neighbors that was correct
                performance_share[level].append(correct_labels/level)

        return {f'Top {level} neighbors':
                {'Share >=1 correct neighbor':np.mean(performance[level]),
                'Share of correct neighbors':np.mean(performance_share[level])}
            for level in test_levels}

    @staticmethod
    def setup_pairwise_loss_tensors(labelsnp):
        """Accepts a list of labels and sets up indexers which can be used
        in a triplet loss function along with whether each pair is a positive or
        negative example.

        Args
        ----
        labelsnp : numpy array 
            Class labels of each node, labelsnp[i] = class of node with intid i

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
        embeddings : pytorch tensor torch.float32
            embeddings to be trained
        labels : numpy array
            Class labels of each node, labelsnp[i] = class of node with intid i"""
        
        batch_relevant_nodes = [i for i,l in enumerate(labels) if not pd.isna(l)]
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
        elif self.distance_metric=='l2':
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
            raise ValueError('distance {} is not implemented'.format(self.distance_metric))

        return loss 
       

    def train(self,epochs,
                    batch_size,
                    test_every_n_epochs = 1,
                    unsupervised = False,
                    learning_rate = 1e-2,
                    fanouts = [10,25],
                    neg_samples = 1,
                    return_intermediate_embeddings = False,
                    test_levels=[5,1]):
        """Trains the network weights to improve the embeddings. Can train via supervised learning with triplet loss,
        semisupervised learning with triplet loss, or fully unsupervised learning.

        Args
        ----
        epochs : int
            number of training passes over the data
        batch_size : int
            number of seed nodes for building the training graph
        test_every_n_epochs : int
            how often to do a full evaluation of the embeddings, expensive for large graphs
        unsupervised : bool
            whether to train completely unsupervised
        learning_rate : float
            learning rate to use in the adam optimizer
        fanouts : list of int
            number of neighbors to sample at each layer for GraphSage
        neg_samples : int
            number of negative samples to use in unsupervised loss
        test_levels : list of ints
            passsed to self.eval, number of neighbors to use for testing accuracy"""

        if not self._masks_set:
            self.set_masks()

        optimizer = th.optim.Adam(it.chain(self.net.parameters(),self.embed.parameters()), lr=learning_rate)

        if not unsupervised:
            sampler = NeighborSampler(self.G, [int(fanout) for fanout in fanouts])
            sampledata = np.nonzero(self.train_mask)[0]
        else:
            sampler = UnsupervisedNeighborSampler(self.G, [int(fanout) for fanout in fanouts],neg_samples)
            sampledata = list(range(len(self.node_ids)))
            unsup_loss_fn = CrossEntropyLoss()
            unsup_loss_fn.to(self.device)

        dataloader = DataLoader(
                            dataset=sampledata,
                            batch_size=batch_size,
                            collate_fn=sampler.sample_blocks,
                            shuffle=True,
                            drop_last=True,
                            num_workers=0)

        

        
        perf = self.evaluate(test_levels=test_levels,test_only=True)

        testtop5, testtop1 = perf['Top 5 neighbors']['Share >=1 correct neighbor'], \
                                perf['Top 1 neighbors']['Share >=1 correct neighbor']

        testtop5tot, testtop1tot = perf['Top 5 neighbors']['Share of correct neighbors'], \
                                perf['Top 1 neighbors']['Share of correct neighbors']

        print(testtop5,testtop1,testtop5tot, testtop1tot)
        print("Test Top5 {:.4f} | Test Top1 {:.4f} | Test Top5 Total {:.4f} | Test Top1 Total {:.4f} ".format(
                testtop5,testtop1,testtop5tot, testtop1tot))

        loss_history = []
        perf_history = [perf]
        if return_intermediate_embeddings:
            all_embeddings = []
            all_embeddings.append(self.embeddings)

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

                    #sup_batch_inputs = self.G.ndata['features'][sup_input_nodes].to(self.device)
                    sup_batch_inputs = self.features[sup_input_nodes].to(self.device)
                    sup_batch_labels = self.labels[sup_seeds]
                    #nodeids = [self.node_ids.loc[self.node_ids.intID==i].id.iloc[0] for i in sup_seeds]

                    #print(sup_batch_labels,nodeids)

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
                self._index = None
                

                print("Epoch {:05d} | Step {:0.1f} | Loss {:.8f}".format(
                        epoch, step, loss.item()))
            if return_intermediate_embeddings:
                all_embeddings.append(self.embeddings)
            loss_history.append(loss.item())
            if epoch % test_every_n_epochs == 0 or epoch==epochs:

                perf = self.evaluate(test_levels=test_levels,test_only=True)

                testtop5, testtop1 = perf['Top 5 neighbors']['Share >=1 correct neighbor'], \
                                        perf['Top 1 neighbors']['Share >=1 correct neighbor']

                testtop5tot, testtop1tot = perf['Top 5 neighbors']['Share of correct neighbors'], \
                                        perf['Top 1 neighbors']['Share of correct neighbors']

                print("Epoch {:05d} | Loss {:.8f} | Test Top5 {:.4f} | Test Top1 {:.4f} | Test Top5 Total {:.4f} | Test Top1 Total {:.4f} ".format(
                        epoch, loss.item(),testtop5,testtop1,testtop5tot, testtop1tot))

                perf_history.append(perf)

        if return_intermediate_embeddings:
            return loss_history,perf_history,all_embeddings     
        else:
            return loss_history,perf_history

    def start_api(self,*args,**kwargs):
        """Launches a fastapi to query this class in its current state."""
        package_path = os.path.dirname(os.path.abspath(__file__))
        production_path = package_path + '/production_model'
        pathlib.Path(production_path).mkdir(exist_ok=True)
        self.save(production_path)
        os.environ['FASTREC_DEPLOY_PATH'] = production_path
        #this import cant be at the top level to prevent circular depedency
        from RecAPI import app
        uvicorn.run(app,*args,**kwargs)


    def save(self, filepath):
        """Save all information neccessary to recover current state of the current instance of
        this object to a folder. Initialization args, graph data, node ids, current trained embedding,
        and current torch paramters are all saved.

        Args
        ----
        filepath : str 
            path on disk to save files"""

        with open(f'{filepath}/dgl.pkl','wb') as pklf:
            pickle.dump(self.G,pklf)

        self.node_ids.to_csv(f'{filepath}/node_ids.csv',index=False)

        with open(f'{filepath}/embed.pkl','wb') as pklf:
            pickle.dump(self.embed,pklf)

        th.save(self.net.state_dict(),f'{filepath}/model_weights.torch')

        with open(f'{filepath}/initargs.pkl','wb') as pklf:
            pickle.dump(self.initargs,pklf)


    @classmethod
    def load(cls, filepath, device=None, faiss_gpu=None):
        """Restore a previous instance of this class from disk.

        Args
        ----
        filepath : str 
            path on disk to load from
        device : str
            optionally override the pytorch device
        faiss_gpu : str
            optionally override whether faiss uses gpu"""

        with open(f'{filepath}/initargs.pkl','rb') as pklf:
            (embedding_dim,
            feature_dim,
            hidden_dim,
            hidden_layers,
            dropout,
            agg_type,
            distance,
            torch_device,
            faiss_gpu_loaded,
            inference_batch_size,
            p_train,
            train_faiss_index) = pickle.load(pklf)

        if device is not None:
            torch_device=device

        if faiss_gpu is not None:
            faiss_gpu_loaded = faiss_gpu

        restored_self = cls(embedding_dim,
                            feature_dim,
                            hidden_dim,
                            hidden_layers,
                            dropout,
                            agg_type,
                            distance,
                            torch_device,
                            faiss_gpu_loaded,
                            inference_batch_size,
                            p_train,
                            train_faiss_index)

        with open(f'{filepath}/dgl.pkl','rb') as pklf:
            restored_self.G = pickle.load(pklf)

        restored_self.node_ids = pd.read_csv(f'{filepath}/node_ids.csv')
        restored_self.node_ids.id = restored_self.node_ids.id.astype(str) 

        with open(f'{filepath}/embed.pkl','rb') as pklf:
            restored_self.embed = pickle.load(pklf)

        restored_self.net.load_state_dict(th.load(f'{filepath}/model_weights.torch',map_location=th.device(torch_device)))

        return restored_self




