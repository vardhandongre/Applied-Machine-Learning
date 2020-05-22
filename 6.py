import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt
import time
from itertools import product
import pickle
import argparse
from tqdm import tqdm

# Vectorize dataset in size 60000 X 100
def vectorize(data):
    num_sample, m, n = data.shape
    vector = data.reshape(num_sample,m*n)
    return vector

# Extract a random sub sample of size n from dataset
def sub_sample(data,n):
    sample_size = np.random.choice(60000,size=n,replace=False)
    sub_data = data[sample_size,:,:]
    return sub_data

def k_means_cluster(n,sub_data,data):
    X = sub_data
    Y = data
    # x_fit = KMeans(n_clusters=n, random_state=0, n_init=50, max_iter=3000, precompute_distances=True, copy_x=False, n_jobs=-1).fit(X)
    x_fit = KMeans(n_clusters=n, random_state=0).fit(X)
    predict = x_fit.predict(Y)
    return predict, x_fit

# Dictionary of 50 clusters ( Level 1 clustering )
def data_dict_one(data,vec,k):
    lev1_dict = {}
    for i in range(k):
        for j in range(len(vec)):
            if vec[j] == i:
                lev1_dict.setdefault(i,[]).append(data[j]) 
    return lev1_dict


# Dictionary of 50 Dictionaries (2500 clusters)
def data_dict_two(data_dict):
    lev2_dict = {}
    cluster_mapping = []
    for i in range(50):
        X = data_dict[i]
        X = np.stack(X, axis=0)
        pred, cluster_model = k_means_cluster(50,X,X)
        cluster_mapping.append(cluster_model)
        level1 = data_dict_one(X,pred,50)
        lev2_dict[i] = level1
    return lev2_dict, cluster_mapping

# Returns the lv1 (out of 50) cluster closest to the patch
def patch_to_lv1_cluster(patch, clus):
    return clus.predict(patch.reshape(1, -1))[0]

# Returns the lv2 (out of 2500) cluster closest to the patch
def patch_to_lv2_cluster(patch, clus, cluster_mapping):
    lv1 = patch_to_lv1_cluster(patch, clus)
    cluster_model = cluster_mapping[lv1]
    temp_lv2 = cluster_model.predict(patch.reshape(1, -1))[0]
    lv2 = 50*lv1 + temp_lv2
    return lv2


class Dataset():
    def __init__(self, folder):
        self.train_feat, self.train_target = self.loadMNIST("train", folder)
        self.test_feat, self.test_target = self.loadMNIST( "t10k", folder )
        self.train_patch16 = self.produce_patches16()
    # Data Loader Function copied from mxmlnkn's reply in https://stackoverflow.com/questions/48257255/how-to-import-pre-downloaded-mnist-dataset-from-a-specific-directory-or-folder
    def loadMNIST(self, prefix, folder):
        intType = np.dtype( 'int32' ).newbyteorder( '>' )
        nMetaDataBytes = 4 * intType.itemsize

        data = np.fromfile( folder + "/" + prefix + '-images-idx3-ubyte', dtype = 'ubyte' )
        magicBytes, nImages, width, height = np.frombuffer( data[:nMetaDataBytes].tobytes(), intType )
        data = data[nMetaDataBytes:].astype( dtype = 'float32' ).reshape( [ nImages, width, height ] )

        labels = np.fromfile( folder + "/" + prefix + '-labels-idx1-ubyte',
                              dtype = 'ubyte' )[2 * intType.itemsize:]
        return data, labels


    def image_to_patches(self, img, add_offsets):
        # img: (28, 28)
        # Returns: patches = (16, 10, 10) patches for add_offsets = False
        # Returns: patches = (144, 10, 10) patches for add_offsets = True
        if stride==6:
            if patch_size==12:
                n_centers = 9
            else:
                n_centers = 16
        elif stride==4:
            n_centers = 25
        elif stride==8:
            n_centers = 9
        if not add_offsets:
            patches = np.zeros((n_centers, patch_size, patch_size))
            for count, (i, j) in enumerate(product(range(0, 20, stride), range(0, 20, stride))):
                patch = img[i:i+patch_size,j:j+patch_size]
                if patch.shape != (patch_size, patch_size):
                    continue
                if count==n_centers:
                    break
                patches[count] = patch
            return patches
        if offset_range==1:
            patches = np.zeros((n_centers*9, patch_size, patch_size))
            pad_img = np.pad(img, 1)
            ranges = range(1, 21, stride)
        elif offset_range==2:
            patches = np.zeros((n_centers*25, patch_size, patch_size))
            pad_img = np.pad(img, 2)
            ranges = range(2, 22, stride)
        count = 0
        for i,j in product(ranges, ranges):
            if offset_range==1:
                coords = [(i-1, j-1), (i, j-1), (i+1, j-1), (i-1, j), (i, j), (i+1, j), (i-1, j+1), (i, j+1), (i+1, j+1)]
            elif offset_range==2:
                coords = [(i-1, j-1), (i, j-1), (i+1, j-1), (i-1, j), (i, j), (i+1, j), (i-1, j+1), (i, j+1), (i+1, j+1),
                (i-2, j-2), (i, j-2), (i+2, j-2), (i-2, j), (i+2, j), (i-2, j+2), (i, j+2), (i+2, j+2),
                (i-2, j-1), (i+2, j-1), (i-2, j+1), (i+2, j+1), (i-1, j-2), (i+1, j-2), (i-1, j+2), (i+1, j+2)]
            for coord in coords:
                patch = pad_img[coord[0]:coord[0]+patch_size,coord[1]:coord[1]+patch_size]
                while patch.shape != (patch_size, patch_size):
                    temp = np.zeros((patch_size))
                    patch = np.transpose(patch)
                    patch = np.concatenate((patch, [temp]), axis=0)
                    patch = np.transpose(patch)
                try:
                    patches[count] = patch
                except:
                    break
                count += 1
        return patches
    # Produces 60,000 patches: 1 for each training image.
    def produce_patches16(self):
        if stride==6:
            if patch_size==12:
                n_centers = 9
            else:
                n_centers = 16
        elif stride==4:
            n_centers = 25
        elif stride==8:
            n_centers = 9
        feat = self.train_feat
        train_patch = np.zeros((feat.shape[0] ,patch_size, patch_size))
        for count, img in enumerate(feat):
            patches = self.image_to_patches(img, add_offsets=False)
            idx = np.random.choice(n_centers)
            patch = patches[idx]
            train_patch[count] = patch
        return train_patch

    def image_to_clusters(self, img, clus, cluster_mapping):
        cluster_list = []
        patches = self.image_to_patches(img, add_offsets=True)
        patches_vec = vectorize(patches)
        for patch in patches_vec:
            cluster_id = patch_to_lv2_cluster(patch, clus, cluster_mapping)
            cluster_list.append(cluster_id)
        return cluster_list

    def image_to_histogram(self, img, clus, cluster_mapping):
        cluster_list = self.image_to_clusters(img, clus, cluster_mapping)
        hist, arr = np.histogram(cluster_list, bins=list(range(2501)))
        return hist



def show_img(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_img(path, img):
    cv2.imwrite(path,img)

def train_and_test_histograms(data, clus, cluster_mapping):
    train = data.train_feat
    test = data.test_feat
    train_list = []
    test_list = []
    print("Pre-processing Training Set")
    for i in tqdm(train):
        # train_list.append(np.concatenate((i.flatten(), data.image_to_histogram(i, clus, cluster_mapping))))
        train_list.append(data.image_to_histogram(i, clus, cluster_mapping))
    print("Preprocessing Test Set")
    for i in tqdm(test):
        # test_list.append(np.concatenate((i.flatten(), data.image_to_histogram(i, clus, cluster_mapping))))
        test_list.append(data.image_to_histogram(i, clus, cluster_mapping))
    return train_list, test_list


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--classifier", help="Specify Classifier Type: rf or xgb", default = 'rf')
    parser.add_argument("-l", "--load_data", help="Load Saved Features", action="store_true")
    parser.add_argument("-o", "--offset_range", help="Specify the Offset which affects the number of centers", default = '1', type=int)
    parser.add_argument("-s", "--stride", help="Specify the Stride when extracting patches", default = '6', type=int)
    parser.add_argument("-p", "--patch_size", help="Specify the dimension of each patch", default = '10', type=int)
    args = parser.parse_args()
    classifier_type = args.classifier
    load_data = args.load_data
    offset_range = args.offset_range
    stride = args.stride
    patch_size = args.patch_size
    print("Loading Data")
    data = Dataset('data')
    print("Loading Data Complete")
    if load_data:
        train_feat = pickle.load(open(f'preprocessing/train_feat_{offset_range}_{stride}_{patch_size}', 'rb'))
        test_feat = pickle.load(open(f'preprocessing/test_feat_{offset_range}_{stride}_{patch_size}', 'rb'))
    else:
        print("Generating Features")
        # Sub-sample to create training set for initial K-means clustering
        patch_subdata = sub_sample(data.train_patch16,6000)
        # Vectorize both datasets
        patch_data_vec = vectorize(data.train_patch16)
        patch_subdata_vec = vectorize(patch_subdata)
        # Perform K-Means clustering for k=50. clus contains the clustering model
        print("Level 1 Clustering")
        clust_vec, clus = k_means_cluster(50,patch_subdata_vec,patch_data_vec)
        # Dict where key = cluster ID and value = list of patches in that cluster
        data_dict1 = data_dict_one(patch_data_vec, clust_vec, 50)
        print("Level 1 Clustering Complete")
        print("Level 2 Clustering")
        data_dict2, cluster_mapping = data_dict_two(data_dict1)
        print("Level 2 Clustering Complete")
        # Generate Train and Test Features
        train_feat, test_feat = train_and_test_histograms(data, clus, cluster_mapping)
        pickle.dump(train_feat, open(f'preprocessing/train_feat_{offset_range}_{stride}_{patch_size}', 'wb'))
        pickle.dump(test_feat, open(f'preprocessing/test_feat_{offset_range}_{stride}_{patch_size}', 'wb'))
    train_feat = np.stack(train_feat, axis=0)
    test_feat = np.stack(test_feat, axis=0)
    if classifier_type=='rf':
        classifier = RandomForestClassifier(n_estimators=100, max_depth=None)
    elif classifier_type=='xgb':
        classifier = XGBClassifier(n_estimators=100, max_depth=None)
    elif classifier_type=='nn':
        classifier = MLPClassifier()
    print(f"Training {classifier_type.upper()} Classifier")
    classifier.fit(train_feat, data.train_target)
    print("Training Completed")
    pred = classifier.predict(test_feat)
    acc = metrics.accuracy_score(data.test_target, pred)
    print(f"Accuracy = {acc}")
    print (f"These results are for offset range = {offset_range}, stride = {stride}, patch_size = {patch_size}.")
