# Applied Machine Learning ( In Progress )  
January 2019 - Present 
  * National Center for Supercomputing Application (Training/Conference/Workshops)
  * Coursework: CS 498 (4 credits)


Repository for Applied Machine Learning Problems

### Some of the content in this repo is a part of coursework please use it without breaking the University Honor Code
 
## 1. Support Vector Machine (R [ No Package/Library ])
A program to train a support vector machine on the data using stochastic gradient descent without the use of any 
library package to train the classifier. Scaled the variables/features so that each has unit variance. 
The code performs a search for an appropriate value of the regularization constant (using validation data)
trying at least the values λ = [1e−3, 1e−2, 1e−1, 1].

## 2. Random Forest Classifier (R, RandomForestClassifier package)
The data has a set of categorical attributes of the mushroom, together with two labels (poisonous or
edible). Using a suitable R package this code builds a random forest to classify a mushroom as edible or poisonous based
on its attributes. It also produces a class-confusion matrix for this problem. 

## 3. Naive Bayes Classifier (Python [ No Package/Library ])
Built and evaluated a naive Bayes classifier that predicts Mathematics scores from all attributes.
For binary attributes, I used a binomial model. For the attributes described as “numeric” which
take a small set of values, I used a Gaussian first then a multinomial model. For the attributes described as “nominal” 
which take a small set of values, I use a multinomial model. Ignored the “absence” attribute.
I have estimated the accuracy by cross-validation. There are 10 folds of data, which exclude 15% of the data at random to serve as test data, The code reports the mean and standard deviation
of the accuracy over these folds.

## 4. Decision Forest For MNIST classification (Python [Pandas,Seaborn, scikit, matplotlib])
Investigated classifying MNIST using a decision forest. Compared the following cases: untouched raw pixels and stretched
bounding box raw pixels. 

## 5. Principal Component Analysis (Python [Seaborn, matplotlib, scipy, scikitlearn])
Investigated the use of principal components to smooth data. Used the traditional 'Iris' dataset. Added a noise to the original data. For noise we added an independent sample from a normal distribution first then in second part a sample from a binomial distribution with a suitable threshold. For each value a plot of the mean-squared error between the original dataset and an expansion onto 1, 2, 3, and 4 principal components was generated to see the effects of PCA. 

## 6. Image Quantization (Python, scikit-learn)
Image Quantization is is a lossy compression technique achieved by compressing a range of values to a single quantum value. Color quantization reduces the number of colors used in an image; this is important for displaying images on devices that support a limited number of colors and for efficiently compressing certain kinds of images. 

## 7. Image Segmentation using Gaussian Mixture Models based on E-M Algorithm (Python)
Image segmentation is an important application of clustering. One breaks an image into k segments, determined by color, texture, etc. These segments are obtained by clustering image pixels by some representation of the image around
the pixel (color, texture, etc.) into k clusters. Then each pixel is assigned to the segment corresponding to its cluster center. Cluster the pixels of a RGB image into 10, 20, and 50 clusters, modelling the pixel values as a mixture of normal distributions and using EM. Display the image obtained by replacing each pixel with the mean of its cluster center.

## 8. Using Variational Inference for Boltzmann Machine to denoise Images (Python, scipy)
Binarize the first 500 images of the MNIST training data by mapping any value below .5 to -1 and any value above to 1. For each image, create a noisy version by randomly flipping 2% of the bits. Now denoise each image using a Boltzmann machine model and mean field inference. Use theta_{ij}=0.2 for the H_i, H_j terms and theta_{ij}=0.5 for the H_i, X_j terms. Report the fraction of all pixels that are correct in the 500 images.

