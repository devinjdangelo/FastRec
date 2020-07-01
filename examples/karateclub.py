import networkx as nx
import pandas as pd
import imageio
import matplotlib.pyplot as plt
import tqdm
import pathlib

from fastrec import GraphRecommender

def animate(labelsnp,all_embeddings,mask):
    labelsnp = labelsnp[mask]

    for i,embedding in enumerate(tqdm.tqdm(all_embeddings)):
        data = embedding[mask]
        fig = plt.figure(dpi=150)
        fig.clf()
        ax = fig.subplots()
        plt.title('Epoch {}'.format(i))

        colormap = ['r' if l=='Administrator' else 'b' for l in labelsnp]
        plt.scatter(data[:,0],data[:,1], c=colormap)

        ax.annotate('Administrator',(data[0,0],data[0,1]))
        ax.annotate('Instructor',(data[33,0],data[33,1]))

        plt.savefig('./ims/{n}.png'.format(n=i))
        plt.close()

    imagep = pathlib.Path('./ims/')
    images = imagep.glob('*.png')
    images = list(images)
    images.sort(key=lambda x : int(str(x).split('/')[-1].split('.')[0]))
    with imageio.get_writer('./animation.gif', mode='I') as writer:
        for image in images:
            data = imageio.imread(image.__str__())
            writer.append_data(data)

if __name__=='__main__':
    g = nx.karate_club_graph()
    nodes = list(g.nodes)
    e1,e2 = zip(*g.edges)
    attributes = pd.read_csv('./karate_attributes.csv')

    sage = GraphRecommender(2,distance='l2')
    sage.add_nodes(nodes)
    sage.add_edges(e1,e2)
    sage.add_edges(e2,e1)
    sage.update_labels(attributes.community)

    epochs, batch_size = 150, 15
    _,_,all_embeddings = sage.train(epochs, batch_size, unsupervised = True, learning_rate=1e-2, 
                            test_every_n_epochs=10, return_intermediate_embeddings=True)

    animate(sage.labels,all_embeddings,sage.entity_mask)

    print(sage.query_neighbors([0,33],k=5))

    sage.start_api()

