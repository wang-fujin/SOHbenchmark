import numpy as np


class Scaler():
    def __init__(self,data):  # data.shape (N,C,L)  or (N,C)
        self.data = data
        if self.data.ndim == 3: # (N,C,L)
            self.mean = self.data.mean(axis=(0,2)).reshape(1,-1,1)
            self.var = self.data.var(axis=(0,2)).reshape(1,-1,1)
            self.max = self.data.max(axis=(0,2)).reshape(1,-1,1)
            self.min = self.data.min(axis=(0,2)).reshape(1,-1,1)
        elif self.data.ndim == 2: # (N,C)
            self.mean = self.data.mean(axis=0).reshape(1, -1)
            self.var = self.data.var(axis=0).reshape(1, -1)
            self.max = self.data.max(axis=0).reshape(1, -1)
            self.min = self.data.min(axis=0).reshape(1, -1)
        else:
            raise ValueError('data dim error!')

    def standerd(self):
        X = (self.data - self.mean) / (self.var + 1e-6)
        return X

    def minmax(self,feature_range=(0,1)):
        if feature_range == (0,1):
            X = (self.data - self.min) / ((self.max - self.min) + 1e-6)
        elif feature_range == (-1,1):
            X = 2*(self.data - self.min) / ((self.max - self.min) + 1e-6)-1
        else:
            raise ValueError('feature_range error!')
        return X

