# QuickRecap
A guide for quick recap of Data Science Interview topics

##### Supervised vs Unsupervised

**Supervised Learning**: one or multiple targets associated with the data.

**Unsupervised Learning**: no target or dependent variable in the data

Target feature can be of two types: 1) Numerical    2) Categorical

**Regression problem** => if target feature is numerical. Eg: House price prediction

**Classification problem** => if target feature is categorical. Eg: Dog vs Cat prediction

Supervised learning could be either a Regression or a Classification problem

Unsupervised => Clustering is one of the approach. Eg: Credit Card Fraudulent detection, Customer Segmentation, Image Clustering, etc

To make sense of **Unsupervised problems**, we use numerous **decomposition techniques** like **PCA**, **t-SNE**, etc

Eg: If we do **t-SNE decomposition** on MNIST (Handwritten digits) dataset, we can seperate the images to some extent just by using 2 components on Image pixels


##### Cross Validation

It is a process of dividing training data into a few parts and perform hyperparameter tuning on such parts

It is also called as **Hold-out set**

**Types of Cross Validations**:

1) K-fold Cross Validation

2) Stratified K-fold Cross Validation

3) Group K-fold Cross Validation

4) Hold-out Cross Validation

5) Leave-one-out Cross Validation


**K-fold CV**: Mainly used for any regression problem. Divide the data into K different sets which are exclusive of each other, each sample/row is assigned with a value between 0 to K-1

**Stratified K-fold CV**: Mainly used for **skewed datasets** and classification problem. It keeps the ratio of labels in each fold as same as the training data in case of Classification problem. In case of Regression, we have to divide the target into **bins**.

Note: No. of bins = 1 + loge(N) where N = no. of samples/rows in your dataset

**Group K-fold CV**: Let's assume a dataset where a patient can have multiple images for skin cancer prediction. Here the patients can be considered as **Groups**

Note: In these kinds of datasets, patient in training data should not appear in validation data

**Hold-out CV**: Mainly used for larger datasets & time-series datasets. Keep one of the fold as hold-out.

Note: We will always calculate loss, accuracy, and other metrics on this hold-out set only

**Leave-one-out CV**: Mainly used for smaller datasets. K = N (no. of samples). We will be training on all data samples except one


