# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:39:20 2020

@author: djdev
"""

import pandas as pd
import geohash2 as gh
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

import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import dgl.nn.pytorch as dglnn

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class NegativeSampler(object):
    def __init__(self, g):
        self.weights = g.in_degrees().float() ** 0.75

    def __call__(self, num_samples):
        return self.weights.multinomial(num_samples, replacement=True)

class UnsupervisedNeighborSampler(object):
    def __init__(self, g, fanouts, num_negs):
        self.g = g
        self.fanouts = fanouts
        self.neg_sampler = NegativeSampler(g)
        self.num_negs = num_negs

    def sample_blocks(self, seed_edges):
        n_edges = len(seed_edges)
        seed_edges = th.LongTensor(np.asarray(seed_edges))
        heads, tails = self.g.find_edges(seed_edges)
        neg_tails = self.neg_sampler(self.num_negs * n_edges)
        neg_heads = heads.view(-1, 1).expand(n_edges, self.num_negs).flatten()

        # Maintain the correspondence between heads, tails and negative tails as two
        # graphs.
        # pos_graph contains the correspondence between each head and its positive tail.
        # neg_graph contains the correspondence between each head and its negative tails.
        # Both pos_graph and neg_graph are first constructed with the same node space as
        # the original graph.  Then they are compacted together with dgl.compact_graphs.
        pos_graph = dgl.graph((heads, tails), num_nodes=self.g.number_of_nodes())
        neg_graph = dgl.graph((neg_heads, neg_tails), num_nodes=self.g.number_of_nodes())
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])

        # Obtain the node IDs being used in either pos_graph or neg_graph.  Since they
        # are compacted together, pos_graph and neg_graph share the same compacted node
        # space.
        seeds = pos_graph.ndata[dgl.NID]
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout, replace=True)
            # Remove all edges between heads and tails, as well as heads and neg_tails.
            _, _, edge_ids = frontier.edge_ids(
                th.cat([heads, tails, neg_heads, neg_tails]),
                th.cat([tails, heads, neg_tails, neg_heads]),
                return_uv=True)
            frontier = dgl.remove_edges(frontier, edge_ids)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)
        return pos_graph, neg_graph, blocks


class NeighborSampler(object):
    def __init__(self, g, fanouts):
        self.g = g
        self.fanouts = fanouts

    def sample_blocks(self, seeds):
        seeds = th.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout, replace=True)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)
        return blocks

class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 agg_type):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, agg_type))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, agg_type))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, agg_type))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.number_of_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        nodes = th.arange(g.number_of_nodes())
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            for start in tqdm.trange(0, len(nodes), batch_size):
                end = start + batch_size
                batch_nodes = nodes[start:end]
                block = dgl.to_block(dgl.in_subgraph(g, batch_nodes), batch_nodes)
                input_nodes = block.srcdata[dgl.NID]

                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[start:end] = h.cpu()

            x = y
        return y

class CrossEntropyLoss(nn.Module):
    def forward(self, block_outputs, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata['h'] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pos_score = pos_graph.edata['score']
        with neg_graph.local_scope():
            neg_graph.ndata['h'] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            neg_score = neg_graph.edata['score']

        score = th.cat([pos_score, neg_score])
        label = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)]).long()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss
    
def load_rw_data(classes_to_load,p_train,max_test):
    """Loads random walk data and converts it into a DGL graph"""


    p = pathlib.Path('./chunks')
    files = p.glob('simchunks*.csv')
    df = pd.DataFrame()
    nclasses = 0
    for f in files:
        if nclasses>=classes_to_load:
            break
        nclasses += 10
        #print(f)
        i = pd.read_csv(f.__str__())
        df= df.append(i)
        #print(len(df))
        
    #df = pd.read_csv('C:\code\geosim\simchunk0.csv')
    df = df.iloc[:,1:]
    df['idxsample'] = df.apply(
        lambda row : str(row['sample']) + '_' + row.id,axis=1)
    df['geohash'] = df.apply(
        lambda row : gh.encode(row.lat,row.lon,precision=7),axis=1)
    
    edges = df[['idxsample','geohash']].groupby(['idxsample','geohash'],
                            as_index=False).size().reset_index(name='counts')

    nodes = edges.idxsample.unique().tolist()
    n_id_nodes = len(nodes)
    nodes = pd.DataFrame(nodes,columns=['Nodes'])
    nodes['classes'] = nodes.Nodes.apply(lambda x : x[1:])
    
    classnums = pd.DataFrame(nodes.classes.unique(),columns=['classes'])
    classnums['label'] = list(range(len(classnums)))
    nodes = nodes.merge(classnums,on='classes')
    
    nodes = nodes.append(pd.DataFrame(edges.geohash.unique().tolist(),columns=['Nodes']))
    nodes['id'] = list(range(len(nodes)))
    
    edges = edges.merge(nodes,left_on='idxsample',right_on='Nodes')
    edges = edges.merge(nodes,left_on='geohash',right_on='Nodes')
    
    n_geohash_nodes = len(nodes) - n_id_nodes

    G = DGLGraph()
    G.add_nodes(len(nodes))
    G.add_edges(edges.id_x.tolist(),edges.id_y.tolist())
    G.add_edges(edges.id_y.tolist(),edges.id_x.tolist())
    G.readonly()

    #ntypes = ['id' for _ in range(n_id_nodes)] + ['geohash' for _ in range(n_geohash_nodes)]
    #etypes = ['to' for _ in range(len(nodes))]
    #G = dgl.to_hetero(G,ntypes,etypes)

    G = dgl.as_heterograph(G)

    edges = edges.merge(nodes,left_on='geohash',right_on='Nodes')
    
    embed = nn.Embedding(len(nodes),256)
    G.ndata['features'] = embed.weight
    labels = [-1 if np.isnan(x) else x for x in nodes.label.tolist()]
    labels = th.tensor(labels,dtype=th.long)
    
    train_mask =  np.random.choice(
        a=[False,True],size=(len(nodes)),p=[1-p_train,p_train])


    node_counts = nodes.groupby('label',as_index=False).size().reset_index(name='counts')
    node_counts = node_counts[node_counts.counts>1]
    test_nodes = node_counts.label.tolist()

    #train_mask = np.array([True] + [False]*5 + [True] + [False]*(len(nodes)-7))
    p_test = 1
    test_labels =  np.random.choice(
        a=list(range(nclasses)),size=min(max_test,nclasses),replace=False)


    test_mask = [l in test_labels and l in test_nodes for l in nodes.label.tolist()]


    is_relevant_node = np.logical_not(nodes.label.isna().to_numpy())
    #test_mask = np.logical_not(is_relevant_node)
    train_mask = np.logical_and(train_mask,is_relevant_node)
    test_mask = np.logical_and(test_mask,is_relevant_node)
    #test_mask = np.logical_or(test_mask,train_mask)

    train_mask = [1 if tf else 0 for tf in train_mask]
    test_mask = [1 if tf else 0 for tf in test_mask]

    return G, embed, labels, train_mask, test_mask,nclasses,is_relevant_node


class SimilarityEmbedder:
    def __init__(self,rw_data,args):

        self.g, self.embed, self.labels, self.train_mask, self.test_mask, self.nclasses, self.is_relevant_mask = rw_data

        self.ptrain = args.p_train
        print(self.g)

        self.embedding_dim = args.embedding_dim
        self.distance = args.distance_metric
        self.device = args.device


        self.nx_g = self.g.to_networkx().to_undirected()
        self.net = SAGE(256, args.num_hidden, args.embedding_dim, args.num_layers, F.relu, args.dropout, args.agg_type)
        self.net.to(self.device)
        self.features = self.embed.weight
        self.features.to(self.device)
        self.embed.to(self.device)
        self.labels.to(self.device)
        self.optimizer = th.optim.Adam(it.chain(self.net.parameters(),self.embed.parameters()), lr=args.lr)

        self.sup_sampler = NeighborSampler(self.g, [int(fanout) for fanout in args.fan_out.split(',')])
        self.unsup_sampler = UnsupervisedNeighborSampler(self.g, [int(fanout) for fanout in args.fan_out.split(',')],args.neg_samples)

        self.train_nid = th.LongTensor(np.nonzero(self.train_mask)[0])
        self.test_nid = th.LongTensor(np.nonzero(self.test_mask)[0])
        self.train_mask = th.BoolTensor(self.train_mask)
        self.test_mask = th.BoolTensor(self.test_mask)

        self.batch_size = args.batch_size
        self.sup_weight = args.sup_weight

        self.unsup_loss = CrossEntropyLoss()
        self.unsup_loss.to(self.device)


        cf = lambda i : (self.sup_sampler.sample_blocks(i), self.unsup_sampler.sample_blocks(i))
        self.dataloader = DataLoader(
                            dataset=self.train_nid.numpy(),
                            batch_size=args.batch_size,
                            collate_fn=cf,
                            shuffle=True,
                            drop_last=True,
                            num_workers=args.num_workers)

        if self.embedding_dim == 2:
            self.all_embeddings = []


        #print('setting up pairwise loss tensors...')
        #self.setup_pairwise_loss_tensors()
        print('setting up pairwise eval tensors...')
        self.setup_pairwise_eval_tensors()
        print('done!')

    def setup_pairwise_loss_tensors(self,labelsnp):
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


    def setup_pairwise_eval_tensors(self):
        test_labels = self.labels[self.test_mask].detach().numpy().tolist()
        labelsnp = self.labels[self.is_relevant_mask].detach().numpy().tolist()

        self.test_idx = []
        self.test_idx2 = []
        for i,test_label in enumerate(tqdm.tqdm(labelsnp)):
            if not self.test_mask[i]:
                continue
            else:
                for j,label in enumerate(labelsnp):
                    if not self.is_relevant_mask[j]:
                        continue
                    if i!=j:
                        self.test_idx.append(i)
                        self.test_idx2.append(j)

        print(len(self.test_idx),len(self.test_idx2))



    def evaluate(self):
        self.net.eval()
        with th.no_grad():
            #loss = pairwise_cosine_distance_loss(embeddings,labels,mask,device)
            if self.distance=='cosine':
                dist = lambda t1,t2 : F.cosine_embedding_loss(t1,
                                                t2,
                                                th.ones(t1.shape[0]).to(self.device),reduce=False)
            elif self.distance=='mse':
                dist = lambda t1,t2 : th.sum(F.mse_loss(t1,t2,reduce=False),dim=1)
            else:
                raise ValueError('distance {} is not implemented'.format(self.distance))

            embeddings = self.net.inference(self.g, self.features,self.batch_size,self.device)[self.is_relevant_mask]

            if self.embedding_dim == 2:
                self.all_embeddings.append(embeddings.detach())


            print('computing embeddings for all nodes...')
            labels = self.labels[self.is_relevant_mask].detach().numpy().tolist()

            print('computing all pairwise distances for test set, keeping closest 5 per label')
            top_5 = {}
            #maximum pairwise distance calculations that can fit in gpu memory
            pairwise_batchsize = int(5e7)
            pbar = tqdm.tqdm(total=len(self.test_idx))
            for batch in range(len(self.test_idx) // pairwise_batchsize + 1):
                startidx, endidx = pairwise_batchsize*batch, min(pairwise_batchsize*(batch+1),len(self.test_idx))
                test_idx_batch = self.test_idx[startidx:endidx]
                test_idx2_batch = self.test_idx2[startidx:endidx]


                test_embeddings = embeddings[test_idx_batch].to(self.device)
                candidate_embeddings = embeddings[test_idx2_batch].to(self.device)
                distances = dist(test_embeddings,candidate_embeddings).detach().cpu().numpy()

                for i,d in enumerate(distances):
                    test_node, candidate_node = test_idx_batch[i], test_idx2_batch[i]
                    test_label,candidate_label = labels[test_node], labels[candidate_node]
                    if test_node not in top_5.keys():
                        top_5[test_node] = [(candidate_node,d)]
                    elif len(top_5[test_node]) < 5:
                        top_5[test_node].append((candidate_node,d))
                        top_5[test_node].sort(key=lambda x : x[1])
                    elif len(top_5[test_node]) == 5:
                        if d < top_5[test_node][-1][1]:
                            top_5[test_node][-1] = (candidate_node,d)
                            top_5[test_node].sort(key=lambda x : x[1])
                    else:
                        raise ValueError('top5 candidate list not <= 5 in length')

                    pbar.update(1)
            pbar.close()

            in_top5 = []
            in_top1 = []
            for test_node,distances in top_5.items():
                in_top5.append(labels[test_node] in [labels[candidate_node] for candidate_node,_ in distances])
                in_top1.append(labels[test_node] == labels[distances[0][0]])

        return np.mean(in_top5), np.mean(in_top1)

    def pairwise_distance_loss(self,embeddings,seeds,labels):
        
        labels = labels.cpu().numpy()
        batch_relevant_nodes = [i for i,l in enumerate(labels) if l!=-1]
        embeddings = embeddings[batch_relevant_nodes]
        labels = labels[batch_relevant_nodes]
        idx1,idx2,target = self.setup_pairwise_loss_tensors(labels)



        losstarget = th.tensor(target).to(self.device)

        if self.distance=='cosine':
            input1 = embeddings[idx1]
            input2 = embeddings[idx2]
            loss = F.cosine_embedding_loss(input1,
                                            input2,
                                            losstarget,
                                            margin=0.5)
        elif self.distance=='mse':
            idx1_pos = [idx for i,idx in enumerate(idx1) if target[i]==1]
            idx1_neg = [idx for i,idx in enumerate(idx1) if target[i]==-1]

            idx2_pos = [idx for i,idx in enumerate(idx2) if target[i]==1]
            idx2_neg = [idx for i,idx in enumerate(idx2) if target[i]==-1]

            input1_pos = embeddings[idx1_pos]
            input2_pos = embeddings[idx2_pos]

            input1_neg = embeddings[idx1_neg]
            input2_neg = embeddings[idx2_neg]

            loss_pos = F.mse_loss(input1_pos,input2_pos)
            loss_neg = th.mean(th.max(th.zeros(input1_neg.shape[0]).cuda(),0.25-th.sum(F.mse_loss(input1_neg,input2_neg,reduce=False),dim=1)))

            loss = loss_pos + loss_neg

        else:
            raise ValueError('distance {} is not implemented'.format(self.distance))

        return loss 

    def train(self,epochs,test_every_n_epochs):

        loss_history = []
        top5_history = []
        top1_history = []


        dur = []
        if self.embedding_dim == 2:
            self.all_embeddings = []
        t = time.time()
        testtop5,testtop1 = self.evaluate()
        print('Eval {}'.format(time.time()-t))

        print("Test Top5 {:.4f} | Test Top1 {:.4f}".format(
                testtop5,testtop1))
        for epoch in range(1,epochs+1):
            
            for step,data in enumerate(self.dataloader):
                sup_blocks, unsupervised_data = data 
                pos_graph, neg_graph, unsup_blocks = unsupervised_data


                self.net.train()

                # these names are confusing because "seeds" are the input
                # to neighbor generation but the output in the sense that we 
                # output their embeddings based on their neighbors...
                # the neighbors are the inputs in the sense that they are what we
                # use to generate the embedding for the seeds.
                if self.sup_weight>0:
                    sup_input_nodes = sup_blocks[0].srcdata[dgl.NID]
                    sup_seeds = sup_blocks[-1].dstdata[dgl.NID]

                    sup_batch_inputs = self.g.ndata['features'][sup_input_nodes].to(self.device)
                    sup_batch_labels = self.labels[sup_seeds].to(self.device)

                    sup_embeddings = self.net(sup_blocks, sup_batch_inputs)

                    sup_loss = self.pairwise_distance_loss(sup_embeddings,sup_seeds,sup_batch_labels)

                if self.sup_weight < 1:
                    unsup_input_nodes = unsup_blocks[0].srcdata[dgl.NID]
                    unsup_seeds = unsup_blocks[-1].dstdata[dgl.NID]

                    unsup_batch_inputs = self.g.ndata['features'][unsup_input_nodes].to(self.device)

                    unsup_embeddings =self.net(unsup_blocks,unsup_batch_inputs)
                    unsup_loss = self.unsup_loss(unsup_embeddings, pos_graph, neg_graph)

                if self.sup_weight==1:
                    loss = sup_loss 
                elif self.sup_weight==0:
                    loss = unsup_loss
                else:
                    loss = self.sup_weight * sup_loss + (1 - self.sup_weight) * unsup_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print("Epoch {:05d} | Step {:0.1f} | Loss {:.8f}".format(
                        epoch, step, loss.item()))
            
            loss_history.append(loss.item())
            if epoch % test_every_n_epochs == 0 or epoch==epochs:
                testtop5,testtop1 = self.evaluate()
                top5_history.append(testtop5)
                top1_history.append(testtop1)

                print("Epoch {:05d} | Loss {:.8f} | Test Top5 {:.4f} | Test Top1 {:.4f}".format(
                        epoch, loss.item(),testtop5,testtop1))                

        self.log_histories(loss_history,top5_history,top1_history,test_every_n_epochs)
        if self.embedding_dim == 2:
            self.animate(test_every_n_epochs)

    def log_histories(self,loss,top5,top1,test_every_n_epochs):
        loss_epochs = list(range(len(loss)))
        test_epochs = list(range(0,len(loss),test_every_n_epochs))

        fig = plt.figure()
        ax = fig.subplots()
        plt.title('Triplet {} Loss by Training Epoch'.format(self.distance))
        fig.text(.5, .05, 'classes {}, batch size {}, embedding dim {}, {} distance, Proportion of labels trained on: {}.'.format(
            self.nclasses,self.batch_size,self.embedding_dim,self.distance,self.ptrain), ha='center')
        plt.scatter(loss_epochs,loss)
        plt.plot(loss_epochs,loss)
        fig.set_size_inches(7, 7, forward=True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig('./Results/loss{}_{}_{}_{}_{}_{}.png'.format(
            self.nclasses,self.batch_size,self.embedding_dim,self.distance,self.ptrain,self.sup_weight))
        plt.close()

        fig = plt.figure()
        ax = fig.subplots()
        plt.title('Test Inference Accuracy'.format(self.distance))
        fig.text(.5, .05, 'classes {}, batch size {}, embedding dim {}, {} distance, Proportion of labels trained on: {}.'.format(
            self.nclasses,self.batch_size,self.embedding_dim,self.distance,self.ptrain), ha='center')
        plt.scatter(test_epochs,top5)
        plt.plot(test_epochs,top5,label='Top 5 Accuracy')
        plt.scatter(test_epochs,top1,label='Top 1 Accuracy')
        plt.legend(loc="lower right")
        plt.plot(test_epochs,top1)
        fig.set_size_inches(7, 7, forward=True)
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.savefig('./Results/topn{}_{}_{}_{}_{}_{}.png'.format(
            self.nclasses,self.batch_size,self.embedding_dim,self.distance,self.ptrain,self.sup_weight))
        plt.close()


    def animate(self,test_every_n_epochs):

        labelsnp = self.labels.detach().numpy().tolist()
        for i,embedding in enumerate(tqdm.tqdm(self.all_embeddings)):
            data = embedding.cpu().numpy()
            fig = plt.figure(dpi=150)
            fig.clf()
            ax = fig.subplots()
            #ax.set_xlim([-1,1])
            #ax.set_ylim([-1,1])
            plt.title('Epoch {}'.format(i*test_every_n_epochs))


            plt.scatter(data[:,0],data[:,1])
            for j,label in enumerate(labelsnp):
                if label==-1:
                    continue
                ax.annotate(label,(data[j,0],data[j,1]))
            #pos = draw(i)  # draw the prediction of the first epoch
            plt.savefig('./ims/{n}.png'.format(n=i))
            plt.close()

        imagep = pathlib.Path('./ims/')
        images = imagep.glob('*.png')
        images = list(images)
        images.sort(key=lambda x : int(str(x).split('/')[-1].split('.')[0]))
        with imageio.get_writer('./Results/training_{}_{}_{}_{}_{}_{}.gif'.format(
            self.nclasses,self.batch_size,self.embedding_dim,self.distance,self.ptrain,self.sup_weight), mode='I') as writer:
            for image in images:
                data = imageio.imread(image.__str__())
                writer.append_data(data)




if __name__=="__main__":

    argparser = argparse.ArgumentParser("GraphSage training")
    argparser.add_argument('--device', type=str, default='cuda',
        help="Device to use for training")
    argparser.add_argument('--num-epochs', type=int, default=400)
    argparser.add_argument('--num-hidden', type=int, default=64)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--agg-type', type=str, default='mean')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--test-every', type=int, default=25)
    argparser.add_argument('--lr', type=float, default=1e-2)
    argparser.add_argument('--sup-weight', type=float, default=1)
    argparser.add_argument('--neg_samples', type=int, default=1)
    argparser.add_argument('--dropout', type=float, default=0)
    argparser.add_argument('--num-workers', type=int, default=0,
        help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--n-classes', type=int, default=10000)
    argparser.add_argument('--p-train', type=float, default=1,
        help="Proportion of labels known at training time")
    argparser.add_argument('--max-test-labels', type=int, default=500,
        help="Maximum number of labels to include in test set, helps with performance \
since currently relying on all pairwise search for testing.")
    argparser.add_argument('--distance-metric', type=str, default='cosine',
        help="Distance metric to use in triplet loss function and nearest neighbors inference, mse or cosine.")
    argparser.add_argument('--embedding-dim',type=int,default=32,help="Dimensionality of the final embedding")
    args = argparser.parse_args()




    print('loading data...')
    rw_data = load_rw_data(args.n_classes,args.p_train,args.max_test_labels)
    trainer = SimilarityEmbedder(rw_data,args)
    trainer.train(args.num_epochs,args.test_every)



    


    #ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=200)
    #ani.save('im.mp4', writer=writer)
