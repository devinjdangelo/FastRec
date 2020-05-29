# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:39:20 2020

@author: djdev
"""

import pandas as pd
import geohash2 as gh
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import networkx as nx
import time
import numpy as np
import pathlib
from math import ceil

from dgl.nn.pytorch import GraphConv
import itertools as it

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from heapq import nsmallest

class GCN(nn.Module):
    def __init__(self,embedding_dim):
        self.embedding_dim = embedding_dim
        super(GCN, self).__init__()
        self.conv1 = GraphConv(512, 256)
        self.conv2 = GraphConv(256, self.embedding_dim)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = th.relu(h)
        h = self.conv2(g, h)
        #if self.embedding_dim == 2:
            #h = th.clamp(h,-1,1)
        return h
    
def load_rw_data(classes_to_load,p_train,max_test):
    p = pathlib.Path('/home/ddangelo/dgl')
    files = p.glob('simchunks*.csv')
    df = pd.DataFrame()
    nclasses = 0
    for f in files:
        if nclasses>=classes_to_load:
            break
        nclasses += 10
        print(f)
        i = pd.read_csv(f.__str__())
        df= df.append(i)
        print(len(df))
        
    #df = pd.read_csv('C:\code\geosim\simchunk0.csv')
    df = df.iloc[:,1:]
    df['idxsample'] = df.apply(
        lambda row : str(row['sample']) + '_' + row.id,axis=1)
    df['geohash'] = df.apply(
        lambda row : gh.encode(row.lat,row.lon,precision=7),axis=1)
    
    edges = df[['idxsample','geohash']].groupby(['idxsample','geohash'],
                            as_index=False).size().reset_index(name='counts')

    nodes = edges.idxsample.unique().tolist()
    nodes = pd.DataFrame(nodes,columns=['Nodes'])
    nodes['classes'] = nodes.Nodes.apply(lambda x : x[1:])
    
    classnums = pd.DataFrame(nodes.classes.unique(),columns=['classes'])
    classnums['label'] = list(range(len(classnums)))
    nodes = nodes.merge(classnums,on='classes')
    
    nodes = nodes.append(pd.DataFrame(edges.geohash.unique().tolist(),columns=['Nodes']))
    nodes['id'] = list(range(len(nodes)))
    
    edges = edges.merge(nodes,left_on='idxsample',right_on='Nodes')
    edges = edges.merge(nodes,left_on='geohash',right_on='Nodes')
    
    G = DGLGraph()
    G.add_nodes(len(nodes))
    G.add_edges(edges.id_x.tolist(),edges.id_y.tolist())
    G.add_edges(edges.id_y.tolist(),edges.id_x.tolist())
    edges = edges.merge(nodes,left_on='geohash',right_on='Nodes')
    
    embed = nn.Embedding(len(nodes),512)
    G.ndata['feat'] = embed.weight
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
    
    return G, embed, labels, train_mask, test_mask,nclasses,is_relevant_node


class SimilarityEmbedder:
    def __init__(self,g, embed, labels, train_mask, 
        test_mask,nclasses,is_relevant_mask,
        device='cuda',distance='cosine',embedding_dim=128):

        self.embedding_dim = embedding_dim
        self.distance = distance
        self.device = device
        self.g = g
        self.embed = embed 
        self.labels = labels
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.is_relevant_mask = is_relevant_mask
        self.nclasses = nclasses 

        self.nx_g = g.to_networkx().to_undirected()
        self.net = GCN(embedding_dim)
        self.net.to(device)
        self.features = embed.weight
        self.features.to(device)
        self.embed.to(device)
        self.labels.to(device)
        self.optimizer = th.optim.Adam(it.chain(self.net.parameters(),embed.parameters()), lr=1e-4)
        print('setting up pairwise loss tensors...')
        self.setup_pairwise_loss_tensors()
        print('setting up pairwise eval tensors...')
        self.setup_pairwise_eval_tensors()
        print('done!')

    def setup_pairwise_loss_tensors(self):
        mask = self.train_mask
        labelsnp = self.labels[mask].detach().numpy().tolist()
        self.idx1 = []
        self.idx2 = []
        self.target = []
        labels_compared = []
        for i,l in enumerate(labelsnp):
            shared_labels = [(i,j) for j,other in enumerate(labelsnp) if other==l and (j,i) not in labels_compared and i!=j]
            unshared_labels = [(i,j) for j,other in enumerate(labelsnp) if other!=l and (j,i) not in labels_compared and i!=j]
            for _,j in shared_labels:
                self.idx1.append(i)
                self.idx2.append(j)
                self.target.append(1)
            for _,j in unshared_labels:
                self.idx1.append(i)
                self.idx2.append(j)
                self.target.append(-1)


            labels_compared += shared_labels

    def setup_pairwise_eval_tensors(self):
        test_labels = self.labels[self.test_mask].detach().numpy().tolist()
        labelsnp = self.labels[self.is_relevant_mask].detach().numpy().tolist()

        self.test_idx = []
        self.test_idx2 = []
        for i,test_label in enumerate(labelsnp):
            if test_label not in test_labels:
                continue
            else:
                for j,label in enumerate(labelsnp):
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
                                                th.ones(t1.shape[0]).cuda(),reduce=False)
            elif self.distance=='mse':
                dist = lambda t1,t2 : th.sum(F.mse_loss(t1,t2,reduce=False),dim=1)
            else:
                raise ValueError('distance {} is not implemented'.format(self.distance))

            embeddings = self.net(self.g, self.features)[self.is_relevant_mask]
            labels = self.labels[self.is_relevant_mask].detach().numpy().tolist()
            test_embeddings = embeddings[self.test_idx]
            candidate_embeddings = embeddings[self.test_idx2]
            distances = dist(test_embeddings,candidate_embeddings).detach().cpu().numpy()

            top_5 = {}
            for i,d in enumerate(distances):
                test_label,candidate_label = labels[self.test_idx[i]], labels[self.test_idx2[i]]
                if test_label not in top_5.keys():
                    top_5[test_label] = [(candidate_label,d)]
                elif len(top_5[test_label]) < 5:
                    top_5[test_label].append((candidate_label,d))
                    top_5[test_label].sort(key=lambda x : x[1])
                elif len(top_5[test_label]) == 5:
                    if d < top_5[test_label][-1][1]:
                        top_5[test_label][-1] = (candidate_label,d)
                        top_5[test_label].sort(key=lambda x : x[1])
                else:
                    raise ValueError('top5 candidate list not <= 5 in length')

            in_top5 = []
            in_top1 = []
            for test_label,distances in top_5.items():
                in_top5.append(test_label in [candidate_label for candidate_label,distance in distances])
                in_top1.append(test_label == distances[0][0])

        #print(top_5)
        #print(in_top5)
        #print(in_top1)
        return np.mean(in_top5), np.mean(in_top1)

    def pairwise_distance_loss(self,embeddings):
        

        #lossinput1 = th.stack(input1)
        #lossinput2 = th.stack(input2)
        losstarget = th.tensor(self.target).cuda()

        if self.distance=='cosine':
            embeddings = embeddings[self.train_mask]
            input1 = embeddings[self.idx1]
            input2 = embeddings[self.idx2]
            loss = F.cosine_embedding_loss(input1,
                                            input2,
                                            losstarget,
                                            margin=0.5)
        elif self.distance=='mse':
            idx1_pos = [idx for i,idx in enumerate(self.idx1) if self.target[i]==1]
            idx1_neg = [idx for i,idx in enumerate(self.idx1) if self.target[i]==-1]

            idx2_pos = [idx for i,idx in enumerate(self.idx2) if self.target[i]==1]
            idx2_neg = [idx for i,idx in enumerate(self.idx2) if self.target[i]==-1]

            embeddings = embeddings[self.train_mask]
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

    def train(self,epochs):

        test_every_n_epochs = 25
        dur = []
        if self.embedding_dim == 2:
            self.all_embeddings = []
        t = time.time()
        testtop5,testtop1 = self.evaluate()
        print('Eval {}'.format(time.time()-t))

        print("Test Top5 {:.4f} | Test Top1 {:.4f}".format(
                testtop5,testtop1))
        for epoch in range(1,epochs+1):
            if epoch >=3:
                t0 = time.time()
            
            self.net.train()
            t = time.time()
            embeddings = self.net(self.g, self.features)
            if self.embedding_dim == 2:
                self.all_embeddings.append(embeddings.detach())


            loss = self.pairwise_distance_loss(embeddings)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
            if epoch >=3:
                dur.append(time.time() - t0)
            
            if epoch % test_every_n_epochs == 0 or epoch==epochs:
                testtop5,testtop1 = self.evaluate()

                print("Epoch {:05d} | Loss {:.8f} | Test Top5 {:.4f} | Test Top1 {:.4f} | Time(s) {:.4f}".format(
                        epoch, loss.item(),testtop5,testtop1, np.mean(dur)))

        if self.embedding_dim == 2:
            self.animate()

    def animate(self):

        labelsnp = self.labels.detach().numpy().tolist()
        for i,embedding in enumerate(self.all_embeddings):
            data = embedding.cpu().numpy()
            fig = plt.figure(dpi=150)
            fig.clf()
            ax = fig.subplots()
            #ax.set_xlim([-1,1])
            #ax.set_ylim([-1,1])


            plt.scatter(data[self.is_relevant_mask,0],data[self.is_relevant_mask,1])
            for j,label in enumerate(labelsnp):
                if label==-1:
                    continue
                ax.annotate(label,(data[j,0],data[j,1]))
            #pos = draw(i)  # draw the prediction of the first epoch
            plt.savefig('./ims/step{n}.png'.format(n=i))
            plt.close()


if __name__=="__main__":
    G, embed, labels, train_mask, test_mask,nclasses,is_relevant_mask = load_rw_data(10000,0.2,500)
    trainer = SimilarityEmbedder(G, embed, labels, train_mask, test_mask,nclasses,is_relevant_mask,distance='cosine',embedding_dim=16)
    trainer.train(200)



    


    #ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=200)
    #ani.save('im.mp4', writer=writer)
