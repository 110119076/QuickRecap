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


## Feature Engineering

If we have useful features, the model will perform better

Mainly depends on the domain knowledge and the data available

Feature Engineering = Creating new features + normalization + transformation

**Date & Time Data**:

Features like

Year    Week of Year    Month    Day of Year    Day of Week    Weekend    Hour    Is Leap Year    Quarter    And many more

**Aggregates**: Using aggregates in pandas, it is quite easy to create features like:

- What’s the month a customer is most active in

- What is the count of cat1, cat2, cat3 for a customer

- What is the count of cat1, cat2, cat3 for a customer for a given week of the year

- What is the mean of num1 for a given customer

Sometimes, for example, when dealing with time-series problems, you might have features which are not individual values but a **list of values** like:

Transactions by a customer in a given period of time

**Binning**:

Feature converting of numbers to categories is known as binning.

When you bin, you can use both the bin and the original feature. Binning also enables you to treat numerical features as categorical.

**Log Transformation**

Another feature that you can create from numerical features is log transformation

If you have a feature in the dataset with high variance compared to other features, then we can **reduce the variance** of that feature using log transformation.

Example: we can apply log(1 + x) to this column to reduce its variance

Sometimes, instead of log, you can also take **exponential**

When you use a log-based evaluation metric like **RMSLE**, in that case you can train on **log-transformed targets** and convert back to original using **exponential on the prediction**

**Handling Missing Values for numerical features**:

1) You can fill them with mean, median or mode

2) Can fill it with a value like 0 or something unique that's already not available in that feature

3) A fancy way of filling missing values is to use **K-Nearest Neighbor method**.

You can select a sample with missing values and find the nearest neighbours utilising some kind of distance metric, for example, **Euclidean distance**. Then you can take the mean of all nearest neighbours and fill up the missing value.

You can use the **KNN imputer implementation** for filling missing values like this.

4) Another way of imputing missing values in a column would be to **train a regression model** that tries to predict missing values in a column based on other columns

Start with one column that has a missing value and treat this column as the target column for regression model without the missing values. Using all the other columns, you now train a model on samples for which there is no missing value in the concerned column and then try to predict target (the same column) for the samples that were removed earlier

**Note**:

For **tree-based models**, imputing values are **unnecessary** as they can handle it themselves.

Always remember to **scale or normalize** your features if you are using **linear models** like logistic regression or a model like SVM. **Tree based models** will work fine **without any normalization** of the features.



## Feature Selection

Having too many features pose a problem well known as the curse of dimensionality

If you have a lot of features, you must also have a lot of training samples to capture all the features

**1) Remove features with very low variance**

It means they are close to being constant and thus, do not add any value to any model at all.

Scikit-learn has an implementation for **VarianceThreshold** that does precisely this

**2) Remove features with high correlation**

For calculating the correlation between different numerical features, you can use the **Pearson correlation**

We can remove one of the highly correlated feature

**3) Univariate ways of feature selection**

Scoring of each feature against a given target

**Mutual information**, **ANOVA F-test** and **chi2** are some of the most popular methods for univariate feature selection

**SelectKBest**: It keeps the top-k scoring features

**SelectPercentile**: It keeps the top features which are in a percentage specified by the user

**Note**: Use chi2 only for data which is **non-negative** in nature. Particularly useful feature selection technique in **NLP** when we have a **bag of words** or **tf-idf** based features

**4) Feature selection using a ML model**

Greedy Feature Selection

Recursive Feature Elimination (RFE)

**Greedy Feature Selection**

Step I: Choose a model

Step II: Select a loss/scoring function

Step III: Iteratively evaluate each feature and add it to the list of good features if it improves loss/score

The computational cost associated with this kind of method is very high.

**Note**: And if you do not use this feature selection properly, then you might even end up **overfitting the model**

**Recursive Feature Elimination (RFE)**

In the previous method, we started with one feature and kept adding new features, but in RFE, we start with all features and **keep removing one feature** in every iteration that provides the least value to a given model

**How do we know which feature offers the least value?**

If we use models like linear SVM or logistic regression, we get a **coefficient** for each feature which decides the **importance of the features**. In case of any tree-based models, we get **feature importance** in place of coefficients.

Remove the feature which has the feature importance or the feature which has a coefficient **close to 0**

Remember that when you use a model like logistic regression for binary classification, the coefficients for features are more positive if they are important for the positive class and more negative if they are important for the negative class

**5) Feature selection using feature importance**

You can fit the model to the data and select features from the model by the feature coefficients or the importance of features. If you use coefficients, you can select a **threshold**, and if the coefficient is above that threshold, you can keep the feature else eliminate it.

You can choose features from one model and use another model to train. For example, you can use Logistic Regression coefficients to select the features and then use Random Forest to train the model on chosen features.

Scikit-learn also offers **SelectFromModel** class that helps you choose features directly from a given model

**6) Feature selection using models that have L1 (Lasso) penalization**

When we have L1 penalization for regularization, most coefficients will be 0 (or close to 0), and we **select the features with non-zero coefficients**

**Note**: All tree-based models provide feature importance

Select features on training data and validate the model on validation data for proper selection of features without overfitting the model





## Hyperparameter Optimization

The parameters that the model has are known as hyper-parameters, i.e. the parameters that control the training/fitting process of the model

If we train a linear regression with SGD, parameters of a model are the slope and the bias and hyperparameter is **learning rate**

**Grid Search**

We specify a grid of parameters. A search over this grid to find the best combination of parameters is known as **grid search**

Example: We can say that n_estimators can be 100, 200, 250, 300, 400, 500; max_depth can be 1, 2, 5, 7, 11, 15 and criterion can be gini or entropy

**Note**: If you have **kfold cross-validation**, you need even more loops which implies even **more time** to find the perfect parameters

We could see our **best k fold accuracy score**, and we could also have the **best parameters** from our grid search

**Random Search**

We randomly select a combination of parameters and calculate the cross-validation score

The time consumed here is less than grid search because we do not evaluate over all different combinations of parameters

We choose how many times we want to evaluate our models, and that’s what decides how much time the search takes

Random search is faster than grid search if the number of iterations is less

**How to optimize for a pipeline?**

Let’s say that we are dealing with a multiclass classification problem. In this problem, the training data consists of two text columns, and you are required to build a model to predict the class.

Let’s assume that the pipeline you choose is to first apply **tf-idf in a semisupervised manner** and then use **SVD with SVM classifier**

We have to select the components of SVD and also need to tune the parameters of SVM in this problem

pipeline.Pipeline(

 [('svd', svd),
 
  ('scl', scl),
  
  ('svm', svm_model)]
  
)

Can still go for **GridSearch** or **RandomSearch** for pipelines as well

Just set the estimator parameter to the pipeline you want to optimize and provide desired ranges/choice of params

Minimization of functions using different kinds of **minimization algorithms**.

This can be achieved by using many minimization functions such as **Downhill Simplex Algorithm**, **Nelder-Mead optimization**, using a **Bayesian technique with Gaussian process** for finding optimal parameters or by using a **Genetic Algorithm**

**How the Gaussian process can be used for hyper-parameter optimization?**

These kind of algorithms need a function to optimize, like **minimization of the function** like we minimize the loss

So, let’s say, you want to find the best parameters for best accuracy and obviously, the more the accuracy is better. Now we cannot minimize the accuracy, but we can minimize it when we multiply it by -1. This way, we are minimizing the negative of accuracy, but in fact, we are maximizing accuracy

Using **gp_minimize** function from scikit-optimize (skopt) library

Define a **parameter space**, **optimization function** that returns negative accuracy here and use it in gp_minimize

result = gp_minimize(

 optimization_function,
 
 dimensions=param_space,
 
 n_calls=15,
 
 n_random_starts=10,
 
 verbose=10
 
)

We can also see (plot) **how we achieved convergence**

from skopt.plots import plot_convergence

plot_convergence(result)


There are many libraries available that offer hyperparameter optimization. **scikit-optimize** is one such library, another useful library is **hyperopt**

hyperopt uses **Tree-structured Parzen Estimator** (**TPE**) to find the most optimal parameters

The ways of tuning hyperparameters described above are the most common, and these will work with almost all models: linear regression, logistic regression, tree-based methods, gradient boosting models such as xgboost, lightgbm, and even neural networks!

**Note**: In gradient boosting, when you **increase the depth**, you should **reduce the learning rate**

**Regularization**

When you create large models or introduce a lot of features, you also make it susceptible to overfitting the training data. To **avoid overfitting**, you need to introduce **noise in training data** features or **penalize the cost function**. This penalization is called regularization and helps with generalizing the model.

In **linear models**, the most common types of regularizations are **L1 and L2**.

**L1** is also known as **Lasso regression** and **L2** as **Ridge regression**

When it comes to **neural networks**, we use **dropouts**, the addition of **augmentations**, **noise**, etc. to regularize our models

**Note**: Using hyper-parameter optimization, you can also **find the correct penalty to use**.



**Approaching text classification/regression**

Let’s say we start with a fundamental task of sentiment classification.

One review maps to one target variable. 

The sentiment score is a combination of score from multiple sentences.

A simple way would be just to create two handmade lists of words. One list will contain all the positive words and another list will include all the negative words.

These lists are also known as **sentiment lexicons**.

If the number of positive words is higher, it is a positive sentiment, and if the number of negative words is higher, it is a sentence with a negative sentiment. Else it's a neutral sentiment.

Splitting a string into a list of words is known as **tokenization**

One of the basic models that you should always try with a classification problem in NLP is **bag of words**

**Bag of Words**

We create a huge sparse matrix that stores counts of all the words in our corpus (corpus = all the documents = all the sentences)

**CountVectorizer** from scikit-learn. The way CountVectorizer works is it first tokenizes the sentence and then assigns a value to each token

So, each token is represented by a unique index. These unique indices are the columns that we see. The CountVectorizer stores this information.

Integrate **word_tokenize** from scikit-learn in CountVectorizer for handling special characters.

Use bag of words with Logistic Regression to predict the sentiment of the sentence.

However, Logistic Regression model took a lot of time to train, let’s see if we can improve the time by using **naïve bayes classifier**.

Naïve bayes classifier is quite popular in NLP tasks as the sparse matrices are huge and naïve bayes is a simple model

**TF-IDF**:

**TF** is **term frequencies**, and **IDF** is **inverse document frequency**.

TF(t) = No. of times term t appears in a doc / Total no. of terms in the doc

IDF(t) = Log( Total no. of docs / No. of docs with term t in it)

TF-IDF(t) = TF(t) * IDF(t)

**TfidfVectorizer** from scikit-learn can be used to calculate this.

Scikit-learn also offers TfidfTransformer. If you have count values, you can use TfidfTransformer and get the same behaviour as TfidfVectorizer.

**N-grams**:

N-grams are combinations of words in order. N-grams are easy to create. You just need to take care of the order.

3 grams of the sentence  "hi, how are you?" are as follows:

[('hi', ',', 'how'),

(',', 'how', 'are'),

('how', 'are', 'you'),

('are', 'you', '?')]

Now, these n-grams become a part of our vocab, and when we calculate counts or tf-idf, we consider **one n-gram as one entirely new token**

So, in a way, we are incorporating context to some extent. Both CountVectorizer and TfidfVectorizer implementations of scikit-learn offers ngrams by ngram_range parameter, which has a minimum and maximum limit.

 **Stemming and lemmatization**

They reduce a word to its smallest form. In the case of stemming, the processed word is called the stemmed word, and in the case of lemmatization, it is known as the lemma.

It must be noted that lemmatization is more aggressive than stemming and stemming is more popular and widely used.

Most common **Snowball Stemmer** and **WordNet Lemmatizer**.

words = ["fishing", "fishes", "fished"]

word=fishing, stemmed_word=fish, lemma=fishing

word=fishes, stemmed_word=fish, lemma=fish

word=fished, stemmed_word=fish, lemma=fished

Stemming: When we do stemming, we are given the smallest form of a word which may or may not be a word in the dictionary for the language the word belongs to.

However, in the case of lemmatization, this will be a word.

**Topic Extraction**

Topic extraction can be done using **non-negative matrix factorization (NMF)** or **latent semantic analysis (LSA)**, which is also popularly known as **singular value decomposition** (**SVD**).

These are decomposition techniques that reduce the data to a given number of components.

**What are stopwords?**

These are high-frequency words that exist in every language. For example, in the English language, these words are “a”, “an”, “the”, “for”, etc.

Removing stopwords is not always a wise choice and depends a lot on the business problem. A sentence like “I need a new dog” after removing stopwords will become “need new dog”, so we don’t know who needs a new dog. We lose a lot of context information if we remove stopwords all the time.

To avoid that, use word embeddings.

**Word Embeddings**

You have seen that till now we converted the tokens into numbers. So, if there are N unique tokens in a given corpus, they can be represented by integers ranging from 0 to N-1.

Now we will represent these integer tokens with vectors. This representation of words into vectors is known as **word embeddings** or **word vectors**.

Google’s **Word2Vec** is one of the oldest approaches to convert words into vectors.

We also have **FastText** from Facebook and **GloVe** (Global Vectors for Word Representation) from Stanford.

The basic idea is to build a shallow network that learns the embeddings for words by reconstruction of an input sentence.

So, you can train a network to predict a missing word by using all the words around and during this process, the network will learn and update embeddings for all the words involved. This approach is also known as **Continuous Bag of Words** or **CBoW model**.

You can also try to take one word and predict the context words instead. This is called **skip-gram model**. Word2Vec can learn embedding using these two methods.

**FastText** learns embeddings for **character n-grams** instead. Just like word n-grams, if we use characters, it is known as character n-grams, and finally, **GloVe** learns these embeddings by using **co-occurrence matrices**.

So, we can say that all these different types of embeddings are in the end returning a dictionary where the **key is a word in the corpus** (for example English Wikipedia) and **value is a vector of size N** (usually 300).

We take all the individual word vectors in a given sentence and create a normalized word vector from all word vectors of the tokens. This provides us with a **sentence vector**.

**Transformer** based networks are able to handle dependencies which are long term in nature. LSTM looks at the next word only when it has seen the previous word. This is not the case with transformers.

It can look at all the words in the whole sentence simultaneously. Due to this, one more advantage is that it can easily be parallelized and uses GPUs more efficiently.

Transformers is a very broad topic, and there are too many models: **BERT, RoBERTa, XLNet, XLM-RoBERTa, T5, etc**.

Please note that these transformers are hungry in terms of computational power needed to train them. Thus, if you do not have a high-end system, it might take much longer to train a model compared to LSTM or TF-IDF based models.

