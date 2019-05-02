import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import itertools


class FrisCommon:
    def nearest_stolp(self, x, stolp_list=None):
        print(x)
        print(stolp_list)
        nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(stolp_list)
        nn = nbrs.kneighbors([x], n_neighbors=1, return_distance=False)
        # print(nn[0][0])
        # print(stolp_list)
        return stolp_list[nn[0][0]]

    def f(self, a, stolp_list=None):
        s_a1 = self.nearest_stolp(a, stolp_list)
        s_a2 = self.nearest_stolp(a, list(filter(lambda x: (x != s_a1).all(), stolp_list)))
        return self.FRIS(a, s_a1, s_a2)  # (self.m(s_a2, a) - self.m(s_a1, a)) / (self.m(s_a2, a) + self.m(s_a1, a))

    def FRIS(self,a, s_a1, s_a2):
        return (self.m(s_a2, a) - self.m(s_a1, a)) / (self.m(s_a2, a) + self.m(s_a1, a))

    def get_nn(self,object, object_list):
        pass
class FrisCluster(FrisCommon):
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
            cur_cluster_stolp = stolp_list[cl].copy()
            stolp_list[cl] = None
            #stolp_list = list(filter(lambda x: (x != cur_cluster_stolp).all(),stolp_list))
            cur_cluster_F = cur_cluster.apply(lambda x: self.sum_f_cluster(x, data, list(filter(lambda x: x is not None, stolp_list))), axis=1)
            maxrow = cur_cluster.loc[cur_cluster_F.idxmax()].values
            print("old stolp: {}. new stolp:{}".format(cur_cluster_stolp,maxrow ))
            stolp_list[cl] = maxrow
        return  stolp_list

    def fit(self, X):
        stolp_list = []
        stolp_list = self.find_stolp(X)
        stolp_list = self.find_stolp(X, stolp_list)
        print(stolp_list)
        for k in range(2,self.__K):
            stolp_list = self.find_stolp(X, stolp_list)
        self.__stolp_list  = stolp_list
        #self.__stolp_list =  self.distribute_stolp_list(stolp_list, X)


    def get_stolp_list(self):
        return self.__stolp_list

    def predict(self, X, Y):
        pass

    def m(self, x, y):
        pass


    def virtual_f(self, a, stolp_list=None):
        s_a1 = self.nearest_stolp(a, stolp_list)
        return ( self.__virtual_m  - self.m(a, s_a1)) / (self.__virtual_m + self.m(a, s_a1))



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



class FrisClass(FrisCommon):
    def __init__(self, F_threshold, alpha=1.5,distance='euclidean'):
        self.__stolp_list = []
        self.__alpha = alpha
        self.__F_threshold =  F_threshold
        if distance == 'euclidean':
            self.distance = distance
            self.m = lambda x, y: np.linalg.norm(x - y)
        self.__nn_stolp = None

    def get_stolp_list(self):
        return  self.__stolp_list

    def set_stolp_list(self, value):
        self.__stolp_list = value
        self.__nn_stolp = NearestNeighbors(metric=self.distance).fit(self.__stolp_list)


    def get_nn(self,object, k):
        nn = self.__nn_stolp.kneighbors([object], n_neighbors=k, return_distance=False)
        return nn[0]
    def get_nn_stolp(self,object, k):
        nn_list = self.get_nn(object,k)
        return list(map(lambda x: self.__stolp_list[x], nn_list))

    def get_stolp(self, stolp_index):
        return self.__stolp_list[stolp_index]

    def get_df_stolps(self,X):
        nn = self.__nn_stolp.kneighbors(X.values, n_neighbors=1, return_distance=False)
        cluster = X.apply(lambda x: nn[x.name][0], axis=1)
        return cluster

    def fit(self, X, Y, stolp_list):
        self.__stolp_list = stolp_list
        self.__nn_stolp = NearestNeighbors(metric=self.distance).fit(stolp_list)
        clusters_list = Y.unique()
        all_cluster_combinations = self.make_pair_combination(clusters_list)
        rival_zone_dict = {}
        for cluster_pair in all_cluster_combinations:
            current_rival_zone = self.get_rival_zone(X[Y == cluster_pair[0]],X[Y == cluster_pair[1]], self.get_stolp(cluster_pair[0]), self.get_stolp(cluster_pair[1]))
            if current_rival_zone.shape[0] != 0:
                rival_zone_dict[cluster_pair] = current_rival_zone





    def get_rival_zone(self,data_Ai, data_Aj, stolp_Ai, stolp_Aj):
        def check_nearest_stolps(a,stolp_Ai, stolp_Aj):
            nn = self.get_nn_stolp(a, 2)
            for i in nn:
                #print(i)
                #print(nn)
                if np.array(i) not in np.array([stolp_Ai, stolp_Aj]):
                    return  False
            return True
        def check_F_metric(a, stolp_list):
            return self.f(a, stolp_list) < self. __F_threshold
        def check_stolp_distance(a, stolp_distance):
            nn_stolp = self.get_nn_stolp(a,1)[0]
            return self.m(a, nn_stolp) < stolp_distance


        cluster_data = pd.concat([data_Ai, data_Aj])
        ns = cluster_data.apply(lambda x: check_nearest_stolps(x,stolp_Ai, stolp_Aj), axis=1)
        F_metric = cluster_data.apply(lambda x: check_F_metric(np.array(x), self.__stolp_list), axis=1)
        stolp_distance = cluster_data.apply(lambda x: check_stolp_distance(np.array(x), self.m(stolp_Ai, stolp_Aj)), axis=1)

        cluster_data["ns"] = ns
        cluster_data["F_metric"] = F_metric
        cluster_data["stolp_distance"] = stolp_distance

        cluster_data["cluster"] = self.get_df_stolps(cluster_data.drop(['ns', 'F_metric', 'stolp_distance'],axis=1))

        return cluster_data

    def call_Dij(self, data):


    def make_pair_combination(self, array):
        return list(filter(lambda x: x[0] != x[1] ,list(itertools.product(array,array))))

class FrisTax:
    pass


