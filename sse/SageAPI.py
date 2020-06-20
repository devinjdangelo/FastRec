import numpy as np
import faiss
import pickle

from geosim import load_rw_data_streaming
from torchmodels import *

from fastapi import FastAPI

app = FastAPI()


@app.on_event("startup")
def startup_event():
    #load graph data
    global G
    global node_ids
    G, _, _, _,_,_,is_relevant_mask, node_ids = load_rw_data_streaming(None,None,None,None,True)
    
    with open('/geosim/embeddings.pkl','rb') as f:
        embeddings = pickle.load(f)

    global index
    global normalized_embeddings
    index = faiss.IndexFlatIP(32)
    normalized_embeddings = np.copy(embeddings)
    #this function operates in place so np.copy any views into a new array before using.
    faiss.normalize_L2(normalized_embeddings)
    index.add(normalized_embeddings)
    
    #D, I = index.search(normalized_search,5+1)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/knn/{hashid}/{k}")
def knn(hashid: str, k: int = 5):
    intid = node_ids.loc[node_ids.id == hashid].intID.iloc[0]
    query = normalized_embeddings[np.newaxis, intid,:]
    D, I = index.search(query,k)
    D = D[0].tolist()
    I = I[0]
    I = [node_ids.loc[node_ids.intID==i].id.iloc[0] for i in I]
    return {"QueryID": hashid, "Nearest Neighbors": I, "Cosine Sim": D} 

@app.get("/knn_jacard/{hashid}/{k}")
def knn(hashid: str, k: int = 5):
    intid = node_ids.loc[node_ids.id == hashid].intID.iloc[0]
    neighbors = np.unique(G.successors(intid).numpy())
    neighbors_2 = [G.successors(i).numpy() for i in neighbors]
    neighbors_2 = [np.unique(n) for n in neighbors_2]
    neighbors_2 = [[node_ids.loc[node_ids.intID == i].id.iloc[0] for i in n] for n in neighbors_2]
    neighbors_2 = np.concatenate(neighbors_2)
    neighbors_2, counts_2 = np.unique(neighbors_2, return_counts=True)
    neighbors = [node_ids.loc[node_ids.intID == i].id.iloc[0] for i in neighbors]

    sortme = list(zip(neighbors_2.tolist(),counts_2.tolist()))
    sortme.sort(key=lambda x : -x[1])
    neighbors_2, counts_2 = zip(*sortme)

    return {"QueryID": hashid, "Nearest Neighbors": neighbors_2,
            "N Shared Locations": counts_2, 'Locations Used': neighbors} 