#Machine Learning and Artificial Intelligence: Homework_1
####Description of the assignment
The purpose of this work is to use some classification algorithms and compare them with a toy dataset given by the sklearn library. 
####Dataset
The Wine dataset is a preset of sklearn, it is a copy  of the UCI ML Wine recognition dataset.
The data is the results of a chemical analysis of wines grown in the same region in Italy by three different cultivators. There are different measurements taken for different constituents found in the three types of wine.
**Characteristics:**
-  178 rows
-  3 classes
- 13 numeric continuos attributes, counting also the class label
-  No missing attribute values
From the 12 attributes we choose to take the first two, which are respectively *Alcohol* and *Malic acid* 
![ ](DataSet.png )
####Training procedure
We split data into Train, Validation and Test with  50%, 20% and 30% partition. 
Then, we train our model on Train set, we tune our hyperparameters on Validation set and finally we evaluate performance on the Test set. 
**Normalization**
Since our dataset is raw and both SVM and K-Nearest Neighboors are critical with respect to distances measurament I decided to give the option to apply the classifiers on normalized data to be able to see the difference between the two scenarios.
For the normalization procedure I've used  `sklearn.preprocessing.StandardScaler` which standardize features by removing the mean and scaling to unit variance fitted on the train set.

$$ z =  (x-u)/s $$

####K-Nearest Neighboors Classifier
The intuition is to classify a new entry calculating the *distance* from this point to its nearest   *k-points* and assigning the label according to the most present label in its k-neighborhood.
So, the critical parameters to choose are *k* and the *distance* metric.
I used for the K-NN algorithm `sklearn.neighbors.KNeighborsClassifier` and trained different model with k = [1,3,5,7] and the minkowski distance metric $$(\sum_{i=1}^{n}{|X_{i} - Y_{i}|^{p}})^{1/p} $$ that in 2-dimensions(p=2) is equal to Euclidean distance.
Choosing *k* is crucial, too little and the classifier is too sensible to outliers, too large and the classifier is not able to generalize enough.
For example with k=1 we can see that the decision boundaries highlight the outlier and compromise the classification.
![ ](knn.png )
![ ](knn_acc.png )
According to our validation process the best value for k is 7 with an accuracy of 80%. 
Therefore, we obtain an accuracy on test set of 81%.

Even if normalizing sometimes could help the classifier in this case it won't. I suppose it's because Alcohol attribute is much more important than Malic acid in distinguish these classes, so normalizing we are balancing the weight of the two attributes and consequencenly we are reducing the accuracy.
![ ](knn_norm.png )
![ ](knn_norm_acc.png )

####SVM-Linear
The intuition here is: if the training data is linearly separable, we can select two parallel hyperplanes that separate the two classes of data, so that the distance between them is as large as possible. The region bounded by these two hyperplanes is called the "margin", and the maximum-margin hyperplane is the hyperplane that lies halfway between them. If we have have more than 2 classes, this operation is performed for each couple. 

Sometimes our data are not linearly separable, so we can define a slack variable that, roughly, indicates how much we must move our point so that it is correctly and confidently classified. When we introduce these variable we must decide how much we penalize these variables, so that our calssification is not falsed too much by the use of them.
The C hyerparameter expresses how much we penalize slack variables, the higher the value the higher is the penalization.
I trained model with different C ([0.001,0.01,0.1,1,10,100,1000]) to see how our dataset change according to this parameter.
I used   `sklearn.svm.SVC ` with kernel = "linear". 
![ ](svmlinear.png )
![ ](svmlinear_acc.png )
The best accuracy on validation set is obtained with C=0.1, in fact using this model on our test set we peak to 85%. 
It is interesting to see that if we choose low values of C our boundaries are wrong because the relaxation are too strong.

For SVM normalization is crucial, because the core is base on distance measument. In fact, with the dataset normalized and C=1 we reach 88% of accuracy.

![ ](svmlinear_norm.png )
![ ](svmlinear_norm_acc.png )

####SVM-RBF
In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces.
The kernel trick avoids the explicit mapping that is needed to get linear learning algorithms to learn a nonlinear function or decision boundary, we can define an arbitrary kernel function k(x~n~,x~m~) that we can substitute for our inner products when we are learning the SVM.
RBF kernel: $$ K(\mathbf {x} ,\mathbf {x’} )=\exp \left(-{\frac {||\mathbf {x} -\mathbf {x’} ||^{2}}{2\sigma ^{2}}}\right) $$
As with the linear SVM we train our model for C = [0.001,0.01,0.1,1,10,100,1000] and gamma set to *scale*.
Gamma is the hyperparameter which express how far the influence of a single training example reaches, with low values meaning ‘far’ and high values meaning ‘close’.
*scale* is equal to $$ gamma = 1 / (n~features~ * X~var~) $$

![ ](svmrbf.png )
![ ](svmrbf_acc.png )

The boundaries are changed, in fact we can clearly obser that with C = 1000, for example, the shape is spheric which is not possible with linear kernel.
The highest accuracy on validation is with C = 100, and this model has a 89% accuracy on test set. 

If we normalize the dataset is easier to see the non-linear boundaries, and with normalization we obtain the highest value of accuracy on validation set. In fact we peak to 83% with  C=1. This model gets 82% on test set.

![ ](svmrbf_norm.png )
![ ](svmrbf_norm_acc.png )

#### GridSearch
When we have more than one hyperparameter to tune is suggested to try each combination.
For example, with RBF kernel both gamma and C have to be tuned in order to reach the best result.
I've trained several models with all the combinations of  C= [0.1, 1, 10, 100, 1000, 10000] and gamma = [0.001, 0.01, 0.1, 1, 10, 100]
![ ](gridmanual_acc.png )
We can get that gamma too low is not able to shape correctly the dataset and gamma too high overfits the model. The best accuracy on validation set is obtained with C=0.1 and gamma=1, this model reaches 83% on test set.

With normalization the best model is with C=1 and gamma=1, but the performace remain the same with 83% of accuracy on test set.
![ ](gridmanual_norm_acc.png )












































 

