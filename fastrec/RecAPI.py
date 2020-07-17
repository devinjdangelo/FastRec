import os

from typing import List
from pydantic import BaseModel
import numpy as np
from fastapi import FastAPI

from GraphSimRec import GraphRecommender

class NodeIdList(BaseModel):
    ids : List[str]

class UnseenNodes(BaseModel):
    nodelist : List[str]
    neighbors : List[List[str]]

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


@app.get("/knn/{nodeid}")
def knn(nodeid: str, k: int = 5, labels : bool=False):
    """Returns the nearest k nodes in the graph using faiss

    Args
    ----
    nodeid : str
        identifier of the query node

    k : int
        number of neighbors to return

    labels : bool
        if true, return the class label for each node in the list of neighbors

    Returns:
    K nearest neighbors, distances, and labels of neighbors"""
    return sage.query_neighbors([nodeid], k, return_labels=labels)

@app.post("/knn/")
def knn_post(nodelist : NodeIdList, k: int = 5, labels : bool=False):
    """Returns the nearest k nodes in the graph using faiss
    Args
    ----
    nodelist : NodeIdList
        identifier of the query nodes

    k : int
        number of neighbors to return

    labels : bool
        if true, return the class label for each node in the list of neighbors

    Returns:
    K nearest neighbors, distances, and labels of neighbors"""
    return sage.query_neighbors(nodelist.ids, k, return_labels=labels)


@app.post('/knn_unseen/')
def knn_unseen(unseen_nodes : UnseenNodes, k: int = 5, labels : bool=False):
    """Returns the k nearest neighbors in the graph for
    query nodes that do not currently exist in the graph. 
    The unseen nodes must exclusively have neighbors that do
    already exist in the graph. We can then estimate their 
    embedding by average the embedding of their neighbors.

    Args
    ----
    unseen_nodes : UnseenNodes
        Contains the ids of the unseen nodes and their neighbors

    k : int
        number of nearest neighbors to query
        
    labels : bool
        if true, return the class label for each node in the list of neighbors

    Returns
    -------
    k nearest neighbors, distances, and labels of neighbors"""

    nodelist, neighbors = unseen_nodes.nodelist, unseen_nodes.neighbors
    embeddings = [np.mean(sage.get_embeddings(nlist),axis=0) for nlist in neighbors]
    embeddings = np.array(embeddings)
    D,I = sage._search_index(embeddings,k)
    I,L = sage._faiss_ids_to_nodeids(I,labels)
    if labels:
        output = {node:{'neighbors':i,'neighbor labels':l,'distances':d.tolist()} for node, d, i, l in zip(nodelist,D,I,L)}
    else:
        output = {node:{'neighbors':i,'distances':d.tolist()} for node, d, i in zip(nodelist,D,I)}

    return output

