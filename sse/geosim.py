# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 20:55:44 2020

@author: djdev
"""

import hashlib
import pandas as pd
import numpy as np
import itertools as it
import pathlib
import tqdm
import geohash2 as gh
import multiprocessing as mp 
from multiprocessing.pool import Pool
import random
import argparse
import pickle

import dgl
from dgl import DGLGraph
from dgl.data.utils import save_graphs

import torch as th
import torch.nn as nn


topLeft = (39.127930,-77.559037)
topRight = (39.117276,-76.847569)
botRight = (38.730519,-76.718615)
botLeft = (38.790486,-77.324926)

minLat = 38.730519
maxLat = 39.127930
minLon = -77.559037
maxLon = -76.718615

DIM = 2

def genSamples(npaths,nsteps):

    startpt = np.random.uniform(minLat,maxLat,size=npaths), \
        np.random.uniform(minLon,maxLon,size=npaths)
    startpt = np.array(startpt).T.reshape(npaths,1,DIM)
    steps = (np.random.rand(npaths,nsteps,DIM)*2 - 1)/4000
    paths = startpt + steps.cumsum(1)
    
    return paths

def downSample(paths,nsamples=2,maxpts=250,minpts=20):
    df = pd.DataFrame(columns=['id','classid','time','lon','lat'])
    for sample in range(nsamples):
        ndpts = (1-np.random.power(5,size=paths.shape[0])) * maxpts + minpts
        #ndpts = [100 for x in range(len(paths))]
        for i,path in enumerate(paths):
            n = int(ndpts[i])
            times = np.random.choice(path.shape[0], int(n), replace=False)
            sampled = path[times]
            x,y = sampled[:,0],sampled[:,1]
            classid = hashlib.sha256(path).hexdigest()
            uid = str(sample) + '_' + str(classid)

            df = df.append(pd.DataFrame(zip(it.repeat(uid),it.repeat(classid),times,x,y),columns=['id','classid','time','lon','lat']))
            
    return df
    
def simchunk(i):
    #print(f'sim chunk {i} of 6000...')
    #need to reset seed to ensure different randomness accross cpus
    maxpaths_periter = 100
    iters_per_file = max(NPATHS//maxpaths_periter,1)
    np.random.seed()
    npaths = min(NPATHS,maxpaths_periter)
    nsteps = 5000
    df = pd.DataFrame(columns=['id','classid','time','lon','lat','geohash'])
    for _ in range(iters_per_file):
        paths = genSamples(npaths,nsteps)
        nsamples = random.randint(2,3)
        dft = downSample(paths,nsamples=nsamples)
        dft['geohash'] = dft.apply(
            lambda row : gh.encode(row.lat,row.lon,precision=7),axis=1)
        df = df.append(dft)
    df.to_csv('/geosim/simchunks{i}.csv'.format(i=i),index=False)

def load_rw_data_streaming(classes_to_load,p_train,max_test,save,load):
    """Loads random walk data and converts it into a DGL graph in streaming
    fashion to minimize RAM usage for large graph creation"""

    if load:
        with open('/geosim/gdata.pkl','rb') as gpkl:
            G, embed, labels, train_mask, test_mask,nclasses,is_relevant_node, node_ids = pickle.load(gpkl)
        print('jacard baseline...')
        baseline_acc(G,labels,is_relevant_node)
        
        return G, embed, labels, train_mask, test_mask,nclasses,is_relevant_node, node_ids


    p = pathlib.Path('/geosim/')
    files = p.glob('simchunks*.csv')
    df = pd.DataFrame()
    nclasses = 0
    G = DGLGraph()
    node_ids = pd.DataFrame(columns=['id','intID','classid'])
    pbar = tqdm.tqdm(total=classes_to_load)
    for f in files:
        if nclasses>=classes_to_load:
            break

        df = pd.read_csv(f.__str__())
        df = df[['id','classid','geohash']]

        entities = df[['id','classid']].drop_duplicates()
        geohashes = pd.DataFrame(zip(df.geohash,it.repeat(-1)),columns=['id','classid']).drop_duplicates()
        nodes = pd.concat([entities, geohashes])
        nodes = nodes.merge(node_ids,on='id',how='left',suffixes=('','_merged'))
        num_new_nodes = int(nodes.intID.isna().sum())
        current_maximum_id = node_ids.intID.max()
        start =(current_maximum_id+1)
        if np.isnan(start):
            start = 0
        end = start + num_new_nodes
        new_nodes = pd.isna(nodes.intID)
        nodes.loc[new_nodes,'intID'] = list(range(start,end))
        update_nodes = nodes[new_nodes]
        update_nodes = update_nodes[['id','intID','classid']]
        node_ids = pd.concat([node_ids,update_nodes])

        edges = df.merge(nodes,left_on='id',right_on='id',how='left')
        edges = edges.merge(nodes,left_on='geohash',right_on='id',how='left',suffixes=('','_geohash'))
        edges = edges[['intID','intID_geohash']]

        G.add_nodes(num_new_nodes)
        G.add_edges(edges.intID.tolist(),edges.intID_geohash.tolist())
        G.add_edges(edges.intID_geohash.tolist(),edges.intID.tolist())


        added = df.classid.nunique()
        nclasses += added
        pbar.update(added)
    pbar.close()


    G.readonly()
    G = dgl.as_heterograph(G)
    print('graph completed... computing train, test split')

    node_ids['intID'] = pd.to_numeric(node_ids['intID'])
    #node_ids = node_ids.sort_values('intID')

    embed = nn.Embedding(len(node_ids),512)
    G.ndata['features'] = embed.weight

    classnums = pd.DataFrame(node_ids.classid.unique(),columns=['classid'])
    classnums['label'] = list(range(len(classnums)))
    classnums.loc[classnums.classid==-1,'label'] = -1
    node_ids = node_ids.merge(classnums,on='classid')
    node_ids = node_ids.sort_values('intID')
    labels = node_ids.label.tolist()
    labels = np.array(labels,dtype=np.float32)
    
    train_mask =  np.random.choice(
        a=[False,True],size=(len(node_ids)),p=[1-p_train,p_train])


    test_labels =  np.random.choice(
        a=list(range(nclasses)),size=min(max_test,nclasses),replace=False)
    test_mask = [l in test_labels for l in node_ids.label.tolist()]

    is_relevant_node = node_ids.label.to_numpy() >= 0 #only -1 is not relevant
    #test_mask = np.logical_not(is_relevant_node)
    train_mask = np.logical_and(train_mask,is_relevant_node)
    test_mask = np.logical_and(test_mask,is_relevant_node)
    #test_mask = np.logical_or(test_mask,train_mask)

    train_mask = [1 if tf else 0 for tf in train_mask]
    test_mask = [1 if tf else 0 for tf in test_mask]

    node_ids = node_ids[['id','intID']]

    if save:
        with open('/geosim/gdata.pkl','wb') as gpkl:
            data = (G,embed, labels, train_mask, test_mask,nclasses,is_relevant_node, node_ids)
            pickle.dump(data,gpkl)

    print('jacard baseline...')
    baseline_acc(G,labels,is_relevant_node)

    return G, embed, labels, train_mask, test_mask,nclasses,is_relevant_node, node_ids

def baseline_acc(G,labels,is_relevant_node):
    tested = 0
    maxtotest = min(10000,np.sum(is_relevant_node))
    top1 = []
    top5 = []
    pbar = tqdm.tqdm(total=maxtotest)
    for intid,l in enumerate(labels):
        if not is_relevant_node[intid]:
            continue

        neighbors = np.unique(G.successors(intid).numpy())
        neighbors_2 = [G.successors(i).numpy() for i in neighbors]
        neighbors_2 = [np.unique(n) for n in neighbors_2]
        neighbors_2 = np.concatenate(neighbors_2)

        neighbors_2, counts_2 = np.unique(neighbors_2, return_counts=True)

        sortme = list(zip(neighbors_2.tolist(),counts_2.tolist()))
        sortme.sort(key=lambda x : -x[1])
        sortme = sortme[:6]
        neighbors_2, counts_2 = zip(*sortme)
        neighbor_labels = [labels[n] for n in neighbors_2]

        top1.append(l == neighbor_labels[1])
        top5.append(l in neighbor_labels[1:])

        pbar.update(1)

        tested+=1
        if tested>=maxtotest:
            break

    pbar.close()
    print(np.mean(top5),np.mean(top1))
    return np.mean(top5), np.mean(top1)




if __name__=="__main__":
    
    argparser = argparse.ArgumentParser("GraphSage training")
    argparser.add_argument('--npaths', type=int, default=10)
    argparser.add_argument('--nfiles', type=int, default=1000)
    args = argparser.parse_args()


    NPATHS = args.npaths 


    pool = Pool(mp.cpu_count())
    for _ in tqdm.tqdm(pool.imap_unordered(simchunk,range(args.nfiles)), total=args.nfiles):
        pass
    pool.close()
    pool.join()
    #with Pool(mp.cpu_count()) as pp:
        #pp.map(simchunk,range(args.nfiles))

    