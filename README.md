# QuickRecap
A guide for quick recap of Data Science Interview topics

## Supervised vs Unsupervised

**Supervised Learning**: one or multiple targets associated with the data.

**Unsupervised Learning**: no target or dependent variable in the data

Target feature can be of two types: 1) Numerical    2) Categorical

**Regression problem** => if target feature is numerical. Eg: House price prediction

**Classification problem** => if target feature is categorical. Eg: Dog vs Cat prediction

Supervised learning could be either a Regression or a Classification problem

Unsupervised => Clustering is one of the approach. Eg: Credit Card Fraudulent detection, Customer Segmentation, Image Clustering, etc

To make sense of **Unsupervised problems**, we use numerous **decomposition techniques** like **PCA**, **t-SNE**, etc

Eg: If we do **t-SNE decomposition** on MNIST (Handwritten digits) dataset, we can seperate the images to some extent just by using 2 components on Image pixels


## Cross Validation

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

## Evaluation Metrics

Supervised Learning Evaluation Metrics:

1) Classification Metrics

2) Regression Metrics

**Classification Metrics**

Accuracy    Precision    Recall    F1 Score    Area under ROC Curve (AUC)    Log loss    Precision at K (P@K)    Mean Avg Precision at K (MAP@K)

**Regression Metrics**

MAE    MSE    RMSE    R_square (Coefficient of Determination)    Adjusted R_square    Root Mean Squared Logarithmic Error (RMSLE)    Mean Percentage Error (MPE)    Mean Absoulte Percentage Error (MAPE)



## Know the terms:

TP = Predicted Positive, Actual Positive

FP = Predicted Positive, Actual Negative => Type I Error

FN = Predicted Negative, Actual Positive => Type II Error

TN = Predicted Negative, Actual Negative

**Accuracy**

If your classification dataset has equally distributed target classes, you can choose accuracy as your evaluation metric for your model

Accuracy = (TP+TN) / (TP+FP+FN+TN)

**Note**: If dataset is skewed never go for accuracy

Let's say you have a validation set with 90% benign images & 10% malignant images, if your model always predicts the given input image as benign, you will get the accuracy score as 90% which is useless. Thus, accuracy doesn't make any sense in case of skewed dataset.


**Precision**

Precision = TP / (TP+FP)

Precison gives the rate at which our model is correct when predicting the positive samples

Example: Let's say a model predicted 80 benign images as benign out of 90 actual benign and 8 malignant out of 10 actual malignant

In this case, TP = 8, FP = 10, FN = 2, TN = 80

Accuracy = 88% (useless)

Precision = 44.4% which means that our model is correct 44.4% times when trying to predict positive samples / malignant images

**Note**: If you want less FPs then you should aim for higher precision


**Recall** / **Sensitivity** / **TPR**

Recall = TP / (TP+FN)

Recall defines the percentage of positive samples our model predicted correctly

Example: For the above example

Recall = 80% which means that our model predicts 80% of positive samples / malignant images correctly

**Note**: If you want less FNs then you should aim for higher recall


## FP or FN ??

Consider the cancer detection problem statement, in that case you always prefer less FNs (More Recall)

FN in cancer detection => Predicted as Normal but Actually has Cancer (very dangerous)

Consider spam message detection problem statement, in that case you always prefer less FPs (More Precision)

FP in spam message detection => Predicted as spam but Actually it is valid message (user can't lose important messages in SPAM folder)

Note: Choosing between FP or FN depends on the problem statement

Both Precision & Recall ranges between 0 & 1. Value closer to 1 is always better

If Precision > Recall => less FPs and more FNs

If Precision < Recall => more FPs and less FNs

**Note**: Precision & Recall values can change drastically based on the **threshold** that is chosen


**Precision-Recall Curve**:

Choose some thresholds and calculate P & R for each threshold. Create a plot between these values, it is called Precision Recall Curve

**Note**: If threshold is too high then FNs will be high (Recall decreases), if threshold is too low then FPs will be high (Precision decreases)


**F1 Score**

F1 = 2*P*R / (P+R)

When both Precision & Recall matters, go for F1 Score

**Note**: In case of skewed datasets go for F1 Score. F1 Score also ranges from 0 to 1, higher F1 score (closer to 1) represents a good model


**ROC (Reciever Operating Characteristic) Curve**

TPR = TP / (TP+FN)    FPR = FP / (FP+TN)

Calculate TPR & FPR for each threshold and plot them as TPR on y-axis and FPR on x-axis to get an ROC curve

It is used to choose the thresholds for predicting the final output (positive or negative). It tells how threshold impacts TPR & FPR

Observe the tradeoff between FP & FN and choose the threshold.

If you don't want too many FPs then go for higher threshold values, else if you don't want too many FNs then go for lower threshold values



**Area under ROC Curve (AUC)**

Area under ROC curve. It is widely used for skewed binary target classification problems

AUC ranges from 0 to 1

If AUC = 0.5 it means the model is just giving some random predictions

AUC closer to 1 represents good model

**Note**: AUC closer to 0 either represents bad model or try inverting the probabilities (if probability of +ve class is set to p, set it to 1-p)

Otherwise if AUC < 0.5, your model is worse than random predictions (bad sign)

Example: AUC = 0.85 represents that if you take 2 random images from the dataset such that one image belongs to +ve class and the other to -ve class, then the positive class image will rank higher than negative class image with a probability of 85%


**Log loss**

In case of binary classification problem, use log loss as an evaluation metric

Log loss = - (y*log(y_pred) + (1-y)*log(1-y_pred))

**Note**: Log loss penalizes very high for an incorrect prediction


Until now we only saw these metrics for a binary class classification problem, we can expand this to multi class classification problem as well

**Macro Averaged Precision**: Calculate P for each class individually & average them

**Micro Averaged Precision**: Calculate class wise TP & FP and use them to calculate the overall Precision

**Weighted Precision**: Same as macro, but just with weights while averaging depending on the number of items in each class

Confusion Matrix is used to represent TP, FP, FN & TN for any classification problem

For binary classification pbm, confusion matrix size = 2x2, for multi class classification pbm, confusion matrix size = NxN (N = no of classes)



Multi Label Classification Problem => Each sample can have one or more classes associated with it. Eg: Predict different objects from a given image

**Metrics for Multi Label Classification Problem**

Precision @ K ( P@K )    Average Precision @ K ( AP@K )    Mean Average Precision @ K ( MAP@K )    Log Loss

**Precision @ K**

If you have list of actual classes & list of predicted classes for a sample, P@K is defined as number of hits in predicted list considering top_K predictions divided by K

**Average Precision @ K**

AP@K = ( P@1 + P@2 + P@3 + ... + P@K ) / K

**Mean Average Precision @ K**

MAP@K = Average( AP@K )

**Note**: P@K, AP@K and MAP@K all range between 0 to 1 with 0 being the worst to 1 being the best

**Log Loss** / **Mean Columnwise Log Loss**

You can convert the targets to binary format & then use log loss for each column. In the end, you can take average of log loss in each column.


**MAE**

MAE = Average ( | y_actual - y_pred | )

**MSE**

MSE = Average ( ( y_actual-y_pred ) ^ 2 )

**RMSE**

RMSE = Sqrt(MSE)

**R_square** / **Coefficient of Determination**

To check how good your model fits the data

R_square closer to 1 means model fits the data quite well, closer to 0 means model isn't that good

R_squared can also be negative when the model just makes absurd predictions

![](https://github.com/110119076/testing/blob/main/rSquared.png)

**Adjusted R_square**

It helps in assessing the impact of the new variables in a regression model. It helps in avoiding Overfitting

![](https://github.com/110119076/testing/blob/main/rSquaredAdjusted.png)

**Squared Logarithmic Error**

SLE = [log( ( 1 + y_actual ) / ( 1 + y_pred ) )]^2

**Mean Squared Logarithmic Error**

MSLE = Average(SLE)

**Root Mean Squared Logarithmic Error**

RMSLE = Sqrt(MSLE)

**Percentage Error**

% error = ((y_actual - y_pred)/y_actual)*100

**Absolute Percentage Error**

APE = (|y_actual - y_pred|/y_actual)*100

**Mean Absolute Percentage Error**

MAPE = Average(APE)



## Handling Categorical Variables

Categorical variables can be classified into 2 major types

1) Nominal

2) Ordinal

**Nominal**: Variables or Features that have 2 or more categories which do not have any kind of order associated with them. Example: Gender (M/F)

**Ordinal**: Variables or Features that have levels or categories with a particular order associated with them. Order is important. Example: Size (S/M/L)

A categorical variable with only 2 categories is called as **binary category**

A categorical variable that are present in cycles is called as **cyclic category**. Example: Days in a Week, Hours in a Day

Computers do not understand text data and thus, we need to convert these categories to numbers. 

1) **Label Encoding**

A simple way of doing this would be to create a dictionary that maps these values to numbers starting from 0 to N-1, where N is the **total number of categories** in a given feature

Note: LabelEncoder from scikitlearn does not handle NaN values

Label Encoding can be used in many **tree based models**:

Decision Trees      Random Forest      Extra Trees      Boosted Trees Model (XGBoost    GBM      LightGBM)

Label Encoding cannot be used in Linear models, Support Vector Machines, or Neural Networks as they expect data to be normalized

For such linear models & others, we can **binarize the data**

**Binarizing the data**: This is just converting the categories to numbers and then converting them to their binary representation

It becomes easy to store lots of binarized variables like this if we store them in a **sparse format**

2) **Sparse Format**

A sparse format is nothing but a representation or way of storing data in memory in which you do not store all the values but only the values that matter. In the case of binary variables described above, all that matters is where we have ones (1s).

One way to represent this matrix only with ones would be some kind of dictionary method in which keys are indices of rows and columns and value is 1

A notation like this will occupy much less memory because it has to store only 1s values which will be less than the dense array

The total size of the **sparse csr matrix** is the sum of three values:

sparse_example.data.nbytes + sparse_example.indptr.nbytes + sparse_example.indices.nbytes

We prefer **sparse arrays** over **dense** whenever we have a **lot of zeros in our features**

3) **One Hot Encoding**

Even though the sparse representation of binarized features takes much less memory than its dense representation, there is another transformation for categorical variables that takes even less memory, which is One Hot Encoding.

One hot encoding is a binary encoding too in the sense that there are only two values, 0s and 1s. However, it must be noted that it’s not a binary representation.

When one-hot encoding, the vector size has to be same as the number of categories we are looking at. Each vector has a 1 and rest all other values are 0s.

There are many ways other than the above mentioned three, like converting categorical variables to numerical variables based on count using **groupby** & **transform** in pandas

One more trick is to create new features from these categorical variables. For example like concatinating 2 columns using underscore (cat1_cat2)

Whenever you get a categorical variables, follow these simple steps:

- fill the NaN values

- convert them to integers by applying label encoding using LabelEncoder of scikit-learn or by using a mapping dictionary

- create one-hot encoding. Yes, you can skip binarization!

- go for modelling

**Handle NaN values in categorical features:**

One way is to drop the NaN values if they are very less in number. Another way of handling NaN values is to treat them as a completely new category.

**Rare categories**:

Categories which appear only a small percentage of the total number of samples are called as Rare Categories

Now, let’s assume that you have deployed this model which uses this column in production and when the model or the project is live, you get a category in this column that is not present in train.  You model pipeline, in this case, will throw an error and there is nothing that you can do about it.

If this is expected, then you must modify your model pipeline and include a new category.

This new category is known as the **“rare”** category. A rare category is a category which is not seen very often and can include many different categories.

You can also try to **“predict”** the **unknown category** by using a **nearest neighbour** model.

Remember, if you predict this category, it will become one of the categories from the training data.

Consider a dataset with 5 features, named f1, f2, f3, f4 and f5. Let's say that f3 feature might assume a new value when it’s seen in the test set or live data.

We can build a simple model that’s trained on all features except “f3”. Thus, you will be creating a model that predicts “f3” when it’s not known or not available in training. It might not give great performance but might be able to handle those missing values in test set or live data.

If you have a fixed test set, you can add your test data to training to know about the categories in a given feature. This is very similar to semi-supervised learning in which you use data which is not available for training to improve your model. This will also take care of rare values that appear very less number of times in training data but are in abundance in test data. Your model will be more robust.

The above approach might look like that the model may overfit. If you design your cross-validation in such a way that it replicates the prediction process when you run your model on test data, then it’s never going to overfit.

**Note**: You must design your validation sets in such a way that it has categories which are “unseen” in the training set.

**Unknown category**

We can treat “NONE” as unknown. So, if during live testing, we get new categories that we have not seen before, we will mark them as “NONE”

Common NLP problems: We always build model based on a fixed vocabulary. Transformer models like BERT are trained on ~30K words for English. So when we have a new word coming in we mark it as **UNK** (unknown).

So, you can either assume that your test data will have the same categories as training or you can introduce a **rare or unknown category** to training to take care of new categories in test data. So, now, when it comes to test data, all the new, unseen categories will be mapped to **“RARE”**, and all missing values will be mapped to **“NONE”**.

**Note**: Please note that we do not need to normalize data when we use tree-based models

Some other approaches: We will take all the categorical columns and create all combinations of degree two and can combine them using underscores or other symbols

**Target Encoding**

Have to be very careful as this might overfit your model

It is a technique in which each category of a feature is mapped to a **mean target value**, but this must always be done in cross validated manner.

It means that the first thing you do is create the folds, and then use those folds to create **target encoding features** for different columns of the data in the same way you fit and predict the model on folds

If you have created 5 folds, you have to create target encoding 5 times such that in the end, you have **encoding** for variables in each fold which are not derived from the same fold. And then when you fit your model, you must use the same folds again.

Target encoding for unseen test data can be derived from the **full training data** or can be an **average of all the 5 folds**.

When we use target encoding, it’s better to use some kind of **smoothing or adding noise** in the **encoded values**. Smoothing introduces some kind of regularization that helps with not overfitting the model.

**Entity Embedding** (For NNs):

In entity embeddings, the categories are represented as vectors. We represent categories by vectors in both binarization and one hot encoding approaches.

We can thus represent them by **vectors with float values** instead to avoid huge matrices that will take long time for training.

- You have an embedding layer for each categorical feature.

- Every category in a column can now be mapped to an embedding.

- You then reshape these embeddings to their dimension to make them flat and then concatenate all the flattened inputs embeddings.

- Then add a bunch of dense layers, an output layer and you are done.

