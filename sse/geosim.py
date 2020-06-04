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
    steps = (np.random.rand(npaths,nsteps,DIM)*2 - 1)/5000
    paths = startpt + steps.cumsum(1)
    
    return paths

def downSample(paths,nsamples=2,maxpts=100):
    df = pd.DataFrame(columns=['id','classid','time','lon','lat'])
    for sample in range(nsamples):
        #ndpts = (1-np.random.power(5,size=paths.shape[0])) * maxpts
        ndpts = [100 for x in range(len(paths))]
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
    np.random.seed()
    npaths = NPATHS
    nsteps = 5000
    paths = genSamples(npaths,nsteps)
    nsamples = random.randint(2,3)
    df = downSample(paths,nsamples=nsamples)
    df['geohash'] = df.apply(
        lambda row : gh.encode(row.lat,row.lon,precision=7),axis=1)
    df.to_csv('/geosim/simchunks{i}.csv'.format(i=i),index=False)

def load_rw_data_streaming(classes_to_load,p_train,max_test,save,load):
    """Loads random walk data and converts it into a DGL graph in streaming
    fashion so minimize RAM usage for large graph creation"""

    if load:
        with open('/geosim/gdata.pkl','rb') as gpkl:
            data = pickle.load(gpkl)
        return data


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

    embed = nn.Embedding(len(node_ids),256)
    G.ndata['features'] = embed.weight

    classnums = pd.DataFrame(node_ids.classid.unique(),columns=['classid'])
    classnums['label'] = list(range(len(classnums)))
    classnums.loc[classnums.classid==-1,'label'] = -1
    node_ids = node_ids.merge(classnums,on='classid')
    node_ids = node_ids.sort_values('intID')
    node_ids.to_csv('./test.csv')
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

    if save:
        with open('/geosim/gdata.pkl','wb') as gpkl:
            data = (G, embed, labels, train_mask, test_mask,nclasses,is_relevant_node)
            pickle.dump(data,gpkl)


    return G, embed, labels, train_mask, test_mask,nclasses,is_relevant_node


def load_rw_data(classes_to_load,p_train,max_test,save,load):
    """Loads random walk data and converts it into a DGL graph"""

    if load:
        with open('../gdata.pkl','rb') as gpkl:
            data = pickle.load(gpkl)
        return data


    p = pathlib.Path('/geosim/')
    files = p.glob('simchunks*.csv')
    df = pd.DataFrame()
    nclasses = 0
    geohash_ids = pd.DataFrame(columns=['Nodes'])
    for f in tqdm.tqdm(files,total=classes_to_load//1000):
        if nclasses>=classes_to_load:
            break
        nclasses += 1000
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

    if save:
        with open('gdata.pkl','wb') as gpkl:
            data = (G, embed, labels, train_mask, test_mask,nclasses,is_relevant_node)
            pickle.dump(data,gpkl)

    return G, embed, labels, train_mask, test_mask,nclasses,is_relevant_node

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

    