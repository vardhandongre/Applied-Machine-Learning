# Applied Machine Learning ( In Progress ) 
Spring 20 \\
Repository for Applied Machine Learning Problems
 
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
which take a small set of values, I use a multinomial model. Ignore the “absence” attribute.
Estimate accuracy by cross-validation. There are 10 folds of data, excluding 15% of the data at random to serve as test data, The code reports the mean and standard deviation
of the accuracy over these folds.

## 4. Decision Forest For MNIST classification (Python [Pandas,Seaborn, sckit, matplotlib])
Investigated classifying MNIST using a decision forest. Compared the following cases: untouched raw pixels and stretched
bounding box raw pixels.
