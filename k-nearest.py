from sklearn.datasets import load_wine
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.preprocessing import StandardScaler

data, y = load_wine(return_X_y=True)

print(data.shape)
print(y.shape)

X = data[:,:2]

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=37)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.28, random_state=37)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

accuracy=[]

k_values = [1,3,5,7]

for n_neighbors in k_values:

    knn = neighbors.KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)
    scaler = StandardScaler()
    scaler.fit(X_train)

    scaler.transform(X_train) # scale train set
    scaler.transform(X_val) # scale validation set with the same parameters

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                         np.arange(y_min, y_max, 0.2))
    
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)


    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i)"
              % (n_neighbors))

    y_pred = knn.predict(X_val)

    accuracy.append(metrics.accuracy_score(y_val, y_pred))

plt.figure()
# naming the x axis 
plt.xlabel('k') 
# naming the y axis 
plt.ylabel('accuracy') 
plt.xticks(k_values)

plt.plot(k_values, accuracy) 
# giving a title to my graph 
plt.title('Accuracy over k') 
  
# function to show the plot 
plt.show() 

k = input("Choose the K which performs better: ")

scaler = StandardScaler()

scaler.fit_transform(X_train_val)

scaler.transform(X_test)

knn_final = neighbors.KNeighborsClassifier(k)

y_pred_final = knn.predict(X_test)

print(metrics.accuracy_score(y_test,y_pred_final))
