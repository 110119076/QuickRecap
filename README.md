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

