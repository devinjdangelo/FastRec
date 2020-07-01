import pandas as pd
import numpy as np

from dgl.data import RedditDataset
from fastrec import GraphRecommender


data = RedditDataset(self_loop=True)
e1, e2 = data.graph.all_edges()
e1, e2 = e1.numpy(), e2.numpy()
nodes = pd.DataFrame(data.labels,dtype=np.int32,columns=['labels'])
del data #free up some memory

sage = GraphRecommender(128, feature_dim=512, hidden_dim=256, 
    torch_device='cuda', faiss_gpu=True, distance='cosine')
sage.add_nodes(nodes.index.to_numpy())
sage.add_edges(e1,e2)
sage.update_labels(nodes.labels)

perf = sage.evaluate(test_levels=[50,25,10,5])
print(perf)

#epochs, batch_size = 100, 1000 
#sage.train(epochs, batch_size, unsupervised = True, learning_rate=1e-2,test_every_n_epochs=10)

print(sage.query_neighbors([0,1000],k=10))

sage.start_api()



