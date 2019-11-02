from sklearn.datasets import load_wine
from sklearn import neighbors, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.preprocessing import StandardScaler

def plotDecisionBoundary(X_train, y_train, model, param, paramname):
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                         np.arange(y_min, y_max, 0.2))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)


    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel("Alcohol")
    plt.ylabel("Malic acid")

    plt.title("Wine classification ("+paramname+" = %i)"
              % (param))


def kNN():
    accuracy=[]

    k_values = [1,3,5,7]

    max = 0

    for n_neighbors in k_values:

        knn = neighbors.KNeighborsClassifier(n_neighbors)
        
        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train_scaled = scaler.transform(X_train) # scale train set
        X_val_scaled = scaler.transform(X_val) # scale validation set with the same parameters

        knn.fit(X_train_scaled, y_train)
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        
        plotDecisionBoundary(X_train_scaled, y_train, knn, n_neighbors, "k")

        y_pred = knn.predict(X_val_scaled)
        acc = metrics.accuracy_score(y_val, y_pred)
        
        if(acc > max):
            best_k = n_neighbors
            max = acc
            best_model = knn
            best_scaler = scaler
        accuracy.append(acc)

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

    print("Best model with k = "+str(best_k)+" and accuracy "+str(max)+" on validation set")

    X_test_scaled = best_scaler.transform(X_test)

    y_pred_final = best_model.predict(X_test_scaled)

    print("Accuracy on test set: "+str(metrics.accuracy_score(y_test,y_pred_final)))

def linearSVM():
    accuracy=[]

    c_values = [0.001,0.01,0.1,1,10,100,1000]

    max = 0

    for c in c_values:

        l_svm = svm.SVC(kernel='linear', C=c)
        
        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train_scaled = scaler.transform(X_train) # scale train set
        X_val_scaled = scaler.transform(X_val) # scale validation set with the same parameters

        l_svm.fit(X_train_scaled, y_train)
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        
        plotDecisionBoundary(X_train_scaled, y_train, l_svm, c, "c")

        y_pred = l_svm.predict(X_val_scaled)
        acc = metrics.accuracy_score(y_val, y_pred)
        
        if(acc > max):
            best_c = c
            max = acc
            best_model = l_svm
            best_scaler = scaler
        accuracy.append(acc)

    plt.figure()
    # naming the x axis 
    plt.xlabel('log(c)') 
    # naming the y axis 
    plt.ylabel('accuracy') 
    plt.xticks(c_values)
    plt.xscale('log')

    plt.plot(c_values, accuracy) 
    # giving a title to my graph 
    plt.title('Accuracy over k') 
    
    # function to show the plot 
    plt.show() 

    print("Best model with c = "+str(best_c)+"and accuracy "+str(max)+" on validation set")

    X_test_scaled = best_scaler.transform(X_test)

    y_pred_final = best_model.predict(X_test_scaled)

    print("Accuracy on test set: "+str(metrics.accuracy_score(y_test,y_pred_final)))


data, y = load_wine(return_X_y=True)

#print(data.shape)
#print(y.shape)

X = data[:,:2]

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.28, random_state=1)

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

answer = input("Which classifier you want to use: (1) kNN (2) Linear SVM: ")

if(answer == "1"):
    kNN()
if(answer == "2"):
    linearSVM()
