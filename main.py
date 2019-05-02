import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

data = pd.read_csv("data/jain.csv",header=None)

X = data[[0,1]]
Y = data[2]
N = data.shape[0]

import FrisLib as FL

#FC = FL.FrisCluster(10)
#
#FC.fit(X)
#stolps = FC.get_stolp_list()
#
#new_stolps = FC.distribute_stolp_list(stolps,X)
#
#
#nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(new_stolps)
#nn = nbrs.kneighbors(X.values, n_neighbors=1, return_distance=False)
#cluster = X.apply(lambda x: nn[x.name][0], axis=1)
#markers = ('s', 'x', '+')
#colors = ('red', 'blue', 'lightgreen','black','yellow','indigo', 'dimgray', 'olive', 'cyan', 'orchid', 'plum', 'sienna', 'wheat', 'azure')
#cmap = ListedColormap(colors)
#cluster_list = list(cluster)
#for idx, cl in enumerate(np.unique(cluster)):
#    X_val_1 = []
#    X_val_2 = []
#    for i in range(N):
#        if cluster_list[i] == cl:
#            X_val_1.append(X.values[i][0])
#            X_val_2.append(X.values[i][1])
#        plt.scatter(x=X_val_1, y=X_val_2, c=cmap(idx), marker='+', label=cl)
#    #for s in stolps:
#    #    plt.scatter(x=s[0],y=s[1])
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.show()


stable_stolps = [np.array([23. ,  5.6]), np.array([31.65,  6.55]), np.array([17.45, 14.15]),
                 np.array([36.3, 10.7]), np.array([26.6,  4.9]),  np.array([21.1,  8. ]),
                 np.array([17.45, 20.75]), np.array([38.85, 15.5 ]),  np.array([21.5, 15.5]), np.array([ 7.1 , 18.65])]

FrisC = FL.FrisClass(0.5)
FrisC.set_stolp_list(stable_stolps)

nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(stable_stolps)
nn = nbrs.kneighbors(X.values, n_neighbors=1, return_distance=False)
cluster = X.apply(lambda x: nn[x.name][0], axis=1)

clusters_list = cluster.unique()

all_cluster_combinations = FrisC.make_pair_combination(clusters_list)
cluster_pair = all_cluster_combinations[0]
FrisC.get_rival_zone(X[cluster == cluster_pair[0]],
                     X[cluster == cluster_pair[1]],
                     FrisC.get_stolp(cluster_pair[0]),
                     FrisC.get_stolp(cluster_pair[1]))