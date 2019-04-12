import pandas as pd
from sklearn.neighbors import NearestNeighbors


class LearnClass:
    pass


class FrisCluster:
    def __init__(self, K, VirtualM=1):
        self.__K = K
        self.__virtual_m = VirtualM
        self.__StolpList = pd.DataFrame()

    def fit(self, X):

        stolp_index = None
        max_f = -1
        for index, row in X.iterrows():
            f = self.sum_virtual_f(row)
            if f > max_f:
                max_f = f
                stolp_index = index
            self.__StolpList = pd.DataFrame()

        a = X.sample(n=1, weights='num_specimen_seen', random_state=1)
        for k in range(self.__K):



    def predict(self, X, Y):
        pass

    def m(self,x,y):
        pass

    def nearest_stolp(self, x, stolp_list=None):
        if stolp_list is None:
            stolp_list = self.__StolpList
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(stolp_list)
        nn = nbrs.kneighbors(x, n_neighbors=1, return_distance=False)
        return self.__StolpList[nn[0]]


    def virtual_f(self,a, stolp_list=None):
        s_a1 = self.nearest_stolp(a, stolp_list)
        return (self.m(a,s_a1) - self.VirtualF) / (self.m(a,s_a1) + self.VirtualF)


    def sum_virtual_f(self, a):
        return a.apply(self.virtual_f).sum()




class FrisClass:
    pass

class FrisTax:
    pass


