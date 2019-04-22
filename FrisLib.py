import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

class LearnClass:
    pass


class FrisCluster:
    def __init__(self, K, distance='euclidean', virtual_M=1):
        self.__K = K
        self.__virtual_m = virtual_M
        self.__stolp_list = []
        if distance == 'euclidean':
            self.m = lambda x, y: np.linalg.norm(x - y)

    def find_stolp(self, X, stolp_list=[]):
        max_f = -100000
        max_index = None
        for index, row in X.iterrows():
            # print(row)
            # print(stolp_list)
            if row.values not in np.array(stolp_list):
                cur_stolp_list = stolp_list + [row.values]
                # print(cur_stolp_list)
                cur_f = self.sum_virtual_f(X, cur_stolp_list)
                if cur_f > max_f:
                    max_f = cur_f
                    max_index = index
        stolp_list.append(X.iloc[max_index].values)
        return stolp_list

    def distribute_stolp_list(self, stolp_list, data):
        nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(stolp_list)
        nn = nbrs.kneighbors(data.values, n_neighbors=1, return_distance=False)
        cluster = data.apply(lambda x: nn[x.name][0], axis=1)
        for cl in cluster.unique():
            cur_cluster = data[data.index.isin(cluster[cluster == cl].index)]
            cur_cluster_stolp = stolp_list[cl]
            stolp_list = list(filter(lambda x: (x != cur_cluster_stolp).all(),stolp_list))
            cur_cluster_F = cur_cluster.apply(lambda x: self.sum_f_cluster(x, cur_cluster,stolp_list),axis=1)
            maxrow = cur_cluster.loc[cur_cluster_F.idxmax()].values
            print("old stolp: {}. new stolp:{}".format(cur_cluster_stolp,maxrow ))
            stolp_list.append(maxrow)
        return  stolp_list

    def fit(self, X):

        stolp_list = self.find_stolp(X)
        stolp_list = self.find_stolp(X, stolp_list)
        print(stolp_list)
        for k in range(2,self.__K):
            stolp_list = self.find_stolp(X, stolp_list)
        self.__stolp_list = stolp_list

    def get_stolp_list(self):
        return self.__stolp_list

    def predict(self, X, Y):
        pass

    def m(self, x, y):
        pass

    def nearest_stolp(self, x, stolp_list=None):
        nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(stolp_list)
        nn = nbrs.kneighbors([x], n_neighbors=1, return_distance=False)
        # print(nn[0][0])
        # print(stolp_list)
        return stolp_list[nn[0][0]]

    def FRIS(self,a, s_a1, s_a2):
        return (self.m(s_a2, a) - self.m(s_a1, a)) / (self.m(s_a2, a) + self.m(s_a1, a))

    def virtual_f(self, a, stolp_list=None):
        s_a1 = self.nearest_stolp(a, stolp_list)
        return ( self.__virtual_m  - self.m(a, s_a1)) / (self.__virtual_m + self.m(a, s_a1))


    def f(self, a, stolp_list=None):
        s_a1 = self.nearest_stolp(a, stolp_list)
        s_a2 = self.nearest_stolp(a, list(filter(lambda x: (x != s_a1).all(),stolp_list)))
        return self.FRIS(a, s_a1, s_a2) #(self.m(s_a2, a) - self.m(s_a1, a)) / (self.m(s_a2, a) + self.m(s_a1, a))

    def sum_virtual_f(self, a, stolp_list):
        M = a.shape[0]
        return a.apply(lambda x: self.virtual_f(x.values, stolp_list), axis=1).sum()/M

    def sum_f_cluster(self, cluster_stolp, cluster_data,stolp_list):
        M = cluster_data.shape[0]
        return cluster_data.apply(lambda x: self.cluster_f(x, cluster_stolp, stolp_list), axis=1).sum()/M

    def cluster_f(self, a, cluster_stolp, stolp_list):
        enemy_stolp = self.nearest_stolp(a,stolp_list)
        return self.FRIS(a, cluster_stolp, enemy_stolp)

    def sum_f(self, a, stolp_list=None):
        M = a.shape[0]
        return a.apply(lambda x: self.f(x.values, stolp_list), axis=1).sum()/M

class FrisClass:
    pass

class FrisTax:
    pass


