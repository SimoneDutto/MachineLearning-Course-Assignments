#Machine Learning and Artificial Intelligence: Homework #1
#### Description of the assignment
The purpose of this work is to use some classification algorithms and compare them with a toy dataset given by the sklearn library.  
#### Dataset
The Wine dataset is a preset of sklearn, it is a copy  of the UCI ML Wine recognition dataset.
The data is the results of a chemical analysis of wines grown in the same region in Italy by three different cultivators. There are different measurements taken for different constituents found in the three types of wine.
**Characteristics:**
-  178 rows
-  3 classes
- 13 numeric continuos attributes, counting also the class label
-  No missing attribute values

####Training procedure####
From the 12 attributes we choose to take the first two, which are respectively *Alcohol* and *Malic acid* and we split data into Train, Validation and Test with  50%, 20% and 30% partition. 
Then, we train our model on Train set, we tune our hyperparameters on Validation set and finally we evaluate performance on the Test set.

**Normalization**
Since our dataset is raw and both SVM and K-Nearest Neighboors are critical with respect to distances measurament I decided to give the option to apply the classifiers on normalized data to be able to see the difference between the two scenarios.
For the normalization procedure I've used  `sklearn.preprocessing.StandardScaler` which standardize features by removing the mean and scaling to unit variance.
$$ z =  (x-u)/s$$
####K-Nearest Neighboors Classifier
The intuition is to classify a new entry calculating the *distance* from this point to its nearest   *k-points* and assigning the label according to the most present label in its k-neighborhood.
So, the critical parameters to choose are *k* and the *distance* metric.
I used 






 

