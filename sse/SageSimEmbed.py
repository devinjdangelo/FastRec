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

from geosim import load_rw_data_streaming
from torchmodels import *

GPU = faiss.StandardGpuResources()

   
class SimilarityEmbedder:
    def __init__(self,distance):
        

        self.distance_metric = distance
        if self.distance_metric == 'cosine':
            self.distance_function = lambda t1,t2 : F.cosine_embedding_loss(t1,
                                                t2,
                                                th.ones(t1.shape[0]).to(self.device),reduce=False)
        elif self.distance_metric == 'mse':
            self.distance_function = lambda t1,t2 : th.sum(F.mse_loss(t1,t2,reduce=False),dim=1)
        else:
            raise ValueError('distance {} is not implemented'.format(self.distance))

        self._embeddings = None 
        self._index = None 

    def get_embeddings(self, batch_size=None):
        """Updates all node embeddings if needed and returns the embeddings
        
        Args
        ----
        batch_size : how many nodes to include per batch, defaults to self.batch_size
        save_dir : which directory to save embeddings to, does not save to disk if None 

        Returns
        -------
        embeddings node x embedding_dim tensor"""

        if self._embeddings is None:
            batch_size = batch_size if batch_size is not None else self.batch_size
            print('computing embeddings for all nodes...')
            with th.no_grad():
                self._embeddings = self.net.inference(self.g, self.features,batch_size,self.device).detach().cpu().numpy()
        return self._embeddings


    def get_index(self,use_gpu=False):
        """Creates a faiss index for similarity searches over the node embeddings

        Args
        ----
        embeddings : the embeddings to add to faiss index
        use_gpu : whethern to store the index on gpu and use gpu for compute

        Returns
        -------
        a faiss index of input embeddings"""

        if self._index is None:
            embeddings = self.get_embeddings()
            if self.distance=='cosine':
                self._index  = faiss.IndexFlatIP(self.embedding_dim)
                normalized_embeddings = np.copy(embeddings)
                #this function operates in place so np.copy any views into a new array before using.
                faiss.normalize_L2(normalized_embeddings)
                embeddings = normalized_embeddings
            elif self.distance=='mse':
                index = faiss.IndexFlatL2(self.embedding_dim)

            if use_gpu:
                index = faiss.index_cpu_to_gpu(GPU, 0, index)

            index.add(embeddings)

        return self._index

    def get_neighbors(self,inputs,k,label_nodes=True):
        """Returns the k nearest neighbors of inputs

        Args
        ----
        inputs : the vectors to search against the faiss index
        k : how many neighbors to lookup
        label_nodes : lookup labels of nodes or just return integer ids

        Returns
        -------
        D, I distance numpy array and neighbors array from faiss"""

        index = self.get_index()
        D, I = index.search(inputs,k)
        if label_nodes:
            #lookup node labels
            pass

        return D,I



    def evaluate(self):
        """Evaluates performance of current embeddings

        Returns
        -------
        P at least 1 correct neighbors are in top5, and top1 respectively"""
        self.net.eval()
        test_labels = self.labels[self.test_mask]
        labels = self.labels[self.is_relevant_mask]

        index = self.get_index()
        test_embeddings = self.get_embeddings()[self.test_mask]
        _, I = self.get_neighbors(test_embeddings,6,label_nodes=False)

        ft1, ft5 = [], []
        for node, neighbors in enumerate(I):
            label = test_labels[node]
            neighbor_labels = [labels[n] for n in neighbors[1:]]
            ft1.append(label==neighbor_labels[0])
            ft5.append(label in neighbor_labels)

        return np.mean(ft5), np.mean(ft1)

    def pairwise_distance_loss(self,embeddings,seeds,labels):
        
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
        #torch.save(self.net.state_dict(),'/geosim/model_data{}.pt'.format('0'))
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
                    sup_batch_labels = self.labels[sup_seeds]

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
                th.cuda.empty_cache()


                print("Epoch {:05d} | Step {:0.1f} | Loss {:.8f} | Mem+Maxmem {:.3f} / {:.3f}".format(
                        epoch, step, loss.item(), th.cuda.memory_allocated()/(1024**3),th.cuda.max_memory_allocated()/(1024**3)))
            
            loss_history.append(loss.item())
            if epoch % test_every_n_epochs == 0 or epoch==epochs:
                testtop5,testtop1 = self.evaluate()
                top5_history.append(testtop5)
                top1_history.append(testtop1)

                print("Epoch {:05d} | Loss {:.8f} | Test Top5 {:.4f} | Test Top1 {:.4f}".format(
                        epoch, loss.item(),testtop5,testtop1))                

                #torch.save(self.net.state_dict(),'/geosim/model_data{}.pt'.format(str(epoch)))

        self.log_histories(loss_history,top5_history,top1_history,test_every_n_epochs)
        if self.embedding_dim == 2:
            self.animate(test_every_n_epochs)

    def log_histories(self,loss,top5,top1,test_every_n_epochs):
        loss_epochs = list(range(len(loss)))
        test_epochs = list(range(0,len(loss),test_every_n_epochs))

        fig = plt.figure()
        ax = fig.subplots()
        plt.title('Triplet {} Loss by Training Epoch'.format(self.distance))
        #fig.text(.5, .05, 'classes {}, batch size {}, embedding dim {}, {} distance, Proportion of labels trained on: {}.'.format(
            #self.nclasses,self.batch_size,self.embedding_dim,self.distance,self.ptrain), ha='center')
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
        #fig.text(.5, .05, 'classes {}, batch size {}, embedding dim {}, {} distance, Proportion of labels trained on: {}.'.format(
            #self.nclasses,self.batch_size,self.embedding_dim,self.distance,self.ptrain), ha='center')
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

        labelsnp = self.labels[self.is_relevant_mask]
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
    argparser.add_argument('--max-test-labels', type=int, default=1000000,
        help="Maximum number of labels to include in test set, helps with performance \
since currently relying on all pairwise search for testing.")
    argparser.add_argument('--distance-metric', type=str, default='cosine',
        help="Distance metric to use in triplet loss function and nearest neighbors inference, mse or cosine.")
    argparser.add_argument('--embedding-dim',type=int,default=32,help="Dimensionality of the final embedding")
    argparser.add_argument('--save',action='store_true')
    argparser.add_argument('--load',action='store_true')
    args = argparser.parse_args()




    print('loading data...')
    rw_data = load_rw_data_streaming(args.n_classes,args.p_train,args.max_test_labels,args.save,args.load)
    trainer = SimilarityEmbedder(rw_data,args)
    trainer.train(args.num_epochs,args.test_every)



    


    #ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=200)
    #ani.save('im.mp4', writer=writer)
