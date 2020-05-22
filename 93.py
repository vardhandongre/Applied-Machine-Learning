from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from PIL import Image
import imageio as io 
import numpy as np
import cv2
import time
import os

  
# Vanila EM on Gaussian Models for Image Segmentation
class GMM_EM():
    def __init__(self,folder,n_models, mode):
        self.name = folder
        self.mode = mode
        self.n_models = n_models
        self.image = cv2.imread(folder)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.img_arr = np.array(self.image, dtype=np.uint8) / 255
        self.height = self.img_arr.shape[0]
        self.width = self.img_arr.shape[1]
        self.channels = self.img_arr.shape[2]
        self.total_pixels = self.height * self.width
        self.img_arr = np.reshape(self.img_arr, (self.total_pixels,self.channels))
        self.mu_init, self.pi_init = self.initialize()
        if mode == 1:
            self.cov_init = np.array([np.eye(self.channels)]*self.n_models)
        if mode == 2:
            self.cov_init = (1/400)*np.array([np.eye(self.channels)]*self.n_models)
        if mode == 3:
            self.cov_init = np.array([np.cov(np.transpose(self.img_arr))]*self.n_models)
            
            
    # Initialization
    def initialize(self):
        clusters = KMeans(n_clusters = self.n_models).fit(self.img_arr)
        labels = clusters.labels_
        # Initial mu
        mu = clusters.cluster_centers_
        # Initial pi weights
        pi = np.zeros(self.n_models)
        for i in range(self.n_models):
            pi[i] = float((labels == i).sum()) / self.total_pixels
        return mu, pi
    
    def e_step(self,mu,pi,cov):
        if np.all(mu) == None:
            mu = self.mu_init
        if np.all(pi) == None:
            pi = self.pi_init
        if np.all(cov) == None:
            cov = self.cov_init
        w = np.zeros((self.total_pixels,self.n_models))
        
        for i in range(self.total_pixels):
            for j in range(self.n_models):
                temp = np.matmul((self.img_arr[i]-mu[j]),np.linalg.inv(cov[j]))
                w[i,j] = np.exp((-0.5)*np.dot(temp,(self.img_arr[i]-mu[j])))*pi[j]
            w[i,:] = w[i,:]/np.sum(w[i,:])
        return w
    
    def m_step(self,w):
        pi = np.zeros((self.n_models))
        dr = np.zeros((self.n_models))
        mu = np.zeros((self.n_models,self.channels))
        for i in range(self.n_models):
            dr[i] = np.sum(w[:,i])
            # Update pi
            pi[i] = dr[i]/self.total_pixels
            for j in range(self.total_pixels):
                mu[i,] += self.img_arr[j,]*w[j,i] 
            # Update mu
            mu[i,] = mu[i,]/dr[i]
        
        return mu, pi
    
    def perform_em(self):
        mu = None
        pi = None
        cov = None
        for i in range(30): #use 20
            if i == 0:
              mu_a = self.mu_init
            else:
              mu_a = mu
            w = self.e_step(mu,pi,cov)
            mu,pi = self.m_step(w)
            mu_b = mu
            check = self.stop(mu_a, mu_b)
            
            # Threshold
            if check:
                print("EM successfully completed in",i," iteration(s)")
                break
        return w,mu,pi
    
    def stop(self,mu_a, mu_b):
        temp = 0
        criteria = False
        for i in range(self.n_models):
            temp += np.sqrt(np.matmul((mu_a[i]-mu_b[i]).T, (mu_a[i]-mu_b[i])))
        temp /= self.n_models
        if temp < 1e-3:
            criteria = True
            
        return criteria

    
    def segmentation(self):
        w,mu,pi = self.perform_em()
        new_img = np.zeros((self.height, self.width, self.channels))
        
        # segment ids 
        cluster_ids = []
        for i in range(self.total_pixels):
            cluster_ids.append(np.argsort(w[i])[self.n_models-1])
        
        # # cluster means
        # means = np.zeros((self.n_models,self.channels))
        # for i in range(self.n_models):
        #     clusters = []
        #     for j in range(self.total_pixels):
        #         if cluster_ids[j] == i:
        #             clusters.append(self.img_arr[j]) 
        #     clusters = np.array(clusters)
        #     means[i] = clusters.mean(axis=0)
            
        # Recreate Image 
        idx = 0
        for i in range(self.height):
            for j in range(self.width):
                new_img[i][j] = mu[cluster_ids[idx]]
                idx += 1
        plt.imshow(new_img)
        plt.savefig("results/"+self.name[0:-4]+str(self.n_models)+"segments"+"_mode_"+ str(self.mode)+".jpg", dpi=400)
        
        # Visualize weights (Part b)
        if self.n_models == 10:
            dir_name = self.name[0:-4]+"_mode_"+ str(self.mode)+"_weights_visualized"
            os.mkdir(dir_name)
            # 10 images 
            for i in range(10):
                pix_vals = w[:,i]
                w_img = np.reshape(pix_vals, (self.height, self.width))
                w_img = np.array(w_img*255, dtype='uint8')
                imageio.imwrite(dir_name+"/"+img_object.name[0:-4]+"_model_"+str(i+1)+"_weights.jpg",w_img)
        return w,mu,pi
    
    
        
if __name__=='__main__':
    image_list = ['RobertMixed03.jpg']
    models = [10,20,50]
    # Performing EM with different Covariance Matrix
    # Mode Values: 
    # 1(default) = Identity Matrix, 
    # 2 = 0.025 x Identity Matrix, 
    # 3 = estimate covariance from pixels
    mode = [1,2,3] 
    for img in image_list:
        for num in models:
            img_object = GMM_EM(img,num,mode[1])
            weights, clust_means, pi = img_object.segmentation()