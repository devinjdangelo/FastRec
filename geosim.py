# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 20:55:44 2020

@author: djdev
"""

import hashlib
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import itertools as it

import multiprocessing as mp 
from multiprocessing.pool import Pool
import random


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
    df = pd.DataFrame(columns=['sample','id','time','lon','lat'])
    for sample in range(nsamples):
        #ndpts = (1-np.random.power(5,size=paths.shape[0])) * maxpts
        ndpts = [100 for x in range(len(paths))]
        for i,path in enumerate(paths):
            n = int(ndpts[i])
            times = np.random.choice(path.shape[0], int(n), replace=False)
            sampled = path[times]
            x,y = sampled[:,0],sampled[:,1]
            idsha = hashlib.sha256(path).hexdigest()
            df = df.append(pd.DataFrame(zip(it.repeat(sample),it.repeat(idsha),times,x,y),columns=['sample','id','time','lon','lat']))
            
    return df
    
def simchunk(i):
    #print(f'sim chunk {i} of 6000...')
    npaths = 10
    nsteps = 5000
    paths = genSamples(npaths,nsteps)
    nsamples = random.randint(1,3)
    df = downSample(paths,nsamples=nsamples)
    df.to_csv('./chunks/simchunks{i}.csv'.format(i=i))

if __name__=="__main__":
    
    with Pool(mp.cpu_count()) as pp:
        pp.map(simchunk,range(2000))

    