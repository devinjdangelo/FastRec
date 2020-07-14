import os

import numpy as np
from fastapi import FastAPI

from GraphSimRec import GraphRecommender

app = FastAPI()


@app.on_event("startup")
def startup_event():

    global sage
    deploy_path = os.environ['FASTREC_DEPLOY_PATH']
    sage = GraphRecommender.load(deploy_path,device='cpu',faiss_gpu=False)
    #force the index to be trained
    sage.train_faiss = True
    sage.index.nprobe = 100
    assert sage.index.is_trained

@app.get("/")
def read_root():
    return {"GraphData": sage.G.__str__()}


@app.get("/knn/{hashid}/{k}")
def knn(hashid: str, k: int = 5):
    """Returns to nearest k nodes in the graph based on cosine distance using faiss
    Args
    ----
    hashid : identifier of the query node
    k : number of neighbors to return
    nrpobe : number of partitions to search

    Returns:
    K nearest neighbors and cosine similarities"""
    return sage.query_neighbors([hashid], k)

@app.get("/knn_jacard/{hashid}/{k}")
def knn_jacard(hashid: str, k: int = 5):
    """Returns the nodes of the same type in a bipartite graph which share
    the k largest number of nodes of the opposite type. 

    Args
    ----
    hashid : identifier of the query node
    k : number of neighbors to return

    Returns
    -------
    K nearest neighbors and number of shared nodes"""
    intid = sage.node_ids.loc[sage.node_ids.id == hashid].intID.iloc[0]
    neighbors = np.unique(sage.G.successors(intid).numpy())
    neighbors_2 = [sage.G.successors(i).numpy() for i in neighbors]
    neighbors_2 = [np.unique(n) for n in neighbors_2]
    neighbors_2 = [[sage.node_ids.loc[sage.node_ids.intID == i].id.iloc[0] for i in n] for n in neighbors_2]
    neighbors_2 = np.concatenate(neighbors_2)
    neighbors_2, counts_2 = np.unique(neighbors_2, return_counts=True)
    neighbors = [sage.node_ids.loc[sage.node_ids.intID == i].id.iloc[0] for i in neighbors]

    sortme = list(zip(neighbors_2.tolist(),counts_2.tolist()))
    sortme.sort(key=lambda x : -x[1])
    sortme = sortme[:k]
    neighbors_2, counts_2 = zip(*sortme)

    return {"QueryID": hashid, "Nearest Neighbors": neighbors_2,
            "N Shared Locations": counts_2, 'Locations Used': neighbors} 