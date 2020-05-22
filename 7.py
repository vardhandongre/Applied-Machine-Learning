#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 12:05:02 2020

@author: Vardhan Dongre
"""

from sklearn.cluster import KMeans
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
import cv2


class color_quantization():
    def __init__(self,folder,n_colors):
        self.image = cv2.imread(folder)
        self.n_colors = n_colors
        
    def vectorize(self):
        m,n,o = self.image.shape
        assert o == 3
        vec_img = np.reshape(self.image,(m*n,o))
        return vec_img
        
    def processing(self):
        print('Processing started ....... ','\n')
        #image = np.array(self.image, dtype=np.float64)/255
        p,q,r = self.image.shape
        print('The original shape of image:',self.image.shape,'\n')
        vectorImage = self.vectorize()
        print("The shape of vectorized image:",vectorImage.shape,'\n')
        sub_sample = shuffle(vectorImage, random_state=0)[:1000]
        print('Performing K-means clustering ......','\n')
        cluster = KMeans(n_clusters=self.n_colors,random_state=0).fit(sub_sample)
        labels = cluster.predict(vectorImage)
        print("Finished K-means clustering .......",'\n')
        return cluster, labels
    
    def recreate_image(self):
        cluster, labels = self.processing()
        a, b ,c = self.image.shape
        centers = cluster.cluster_centers_
        d = centers.shape[1]
        img = np.zeros((a,b,d))
        idx = 0
        for i in range(a):
            for j in range(b):
                img[i][j] = centers[labels[idx]]
                idx += 1
        return img
        
    def visualization(self):
        fig = plt.figure(figsize=(10,12))
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)
        ax1.imshow(self.image)
        ax1.set_title("Original Image")
        fig.suptitle("Demonstrating Image Quantization using K-Means Clustering",fontsize=16)
        quant_image = self.recreate_image()
        ax2.imshow(quant_image.astype('uint8'))
        ax2.set_title("Quantized Image")
        fig.tight_layout()
        fig.subplots_adjust(top=1.5)
       
        
if __name__ =='__main__':
    data = color_quantization('smallsunset.jpg',64)
    data.visualization()
