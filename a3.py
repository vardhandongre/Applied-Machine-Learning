import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from scipy.stats import binom
import time
import matplotlib.pyplot as plt

class Dataset():
    def __init__(self, fname):
        col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        self.df = pd.read_csv(fname, header=0, names=col_names)
        reduced_col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        self.df = self.df[reduced_col_names]
        self.norm_df = self.df
        self.project_df = self.df
    def normalize_with_normal(self, norm_val):
        self.norm_df = self.df.applymap(lambda x: x+np.random.normal(loc=0.0, scale=norm_val))
    def reshape(self):
        self.project_df = self.pca.inverse_transform(self.project_df)
    def perform_pca(self, dims):
        self.pca = PCA(n_components=dims)
        self.pca.fit(self.norm_df)
        self.project_df = self.pca.transform(self.norm_df)
    def mse(self):
        error = mean_squared_error(self.df, self.project_df)
        return error
    def normalize_with_masking(self, w):
        thres = 1 - w/600
        mask_mat = np.random.binomial(n=1, p=thres, size=[len(self.df), 4])
        self.norm_df = self.df
        self.norm_df = self.norm_df.mul(mask_mat)
        
def part_1():
    data = Dataset('iris.data')
    dim_list = [1,2,3,4]
    for val in [0.1,0.2,0.5,1.0]:
        mse_list = []
        data.normalize_with_normal(val)
        for dim in dim_list:
            data.perform_pca(dim)
            data.reshape()
            mse = data.mse()
            print(val, dim, mse)
            mse_list.append(mse)
        plot(dim_list, mse_list, val, 1)
        print("\n")

def part_2():
    data = Dataset('iris.data')
    dim_list = [1,2,3,4]
    for val in [10,20,30,40]:
        mse_list = []
        data.normalize_with_masking(val)
        for dim in dim_list:
            data.perform_pca(dim)
            data.reshape()
            mse = data.mse()
            print(val, dim, mse)
            mse_list.append(mse)
        plot(dim_list, mse_list, val, 2)
        print('\n')

def plot(x, y, noise, part=1): # part = 1 or 2
    plt.plot(x, y)
    plt.xlabel('Dimension Value')
    plt.ylabel('MSE')
    plt.title(f"MSE vs Dimension Value for Noise = {noise} in Part {part}")
    plt.savefig(f'plot_{noise}_{part}.png')
    plt.close()

if __name__ == "__main__":
    np.random.seed(int(time.time()))
    part_1()
    part_2()
