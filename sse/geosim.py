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

    