from sklearn import neighbors, svm, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def plotDecisionBoundary(X_train, y_train, model, param, paramname, subplt):
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2),
                         np.arange(y_min, y_max, 0.2))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    subplt.contourf(xx, yy, Z, alpha=0.8)
   

    # Plot also the training points
    subplt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                edgecolor='k', s=20)
    subplt.axis(x_max = xx.max(), x_min = xx.min, y_max = yy.max(), y_min = yy.min())
    subplt.set_xlabel("Alcohol")
    subplt.set_ylabel("Malic acid")
    subplt.set_title("Wine classification ("+paramname+" = "+str(param)+")", pad=10)


def kNN(normalize, show):
    accuracy=[]

    k_values = [1,3,5,7]

    max = 0
    scaler = StandardScaler()
    
    if(normalize):
        scaler.fit(X_train)
        X_train_used = scaler.transform(X_train) # scale train set
        X_val_used = scaler.transform(X_val) # scale validation set with the same parameters
    else:
        X_train_used = X_train
        X_val_used = X_val
    
    figure, axs = plt.subplots(2,2, figsize=(20,25), constrained_layout=True)
    axs = axs.ravel()
    i = 0

    for n_neighbors in k_values:

        knn = neighbors.KNeighborsClassifier(n_neighbors)
        knn.fit(X_train_used, y_train)
        
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if show:
            plotDecisionBoundary(X_train_used, y_train, knn, n_neighbors, "k", axs[i])

        y_pred = knn.predict(X_val_used)
        acc = metrics.accuracy_score(y_val, y_pred)
        
        if(acc > max):
            best_k = n_neighbors
            max = acc
            best_model = knn
        accuracy.append(acc)
        i+=1
    plt.figure()
    # naming the x axis 
    plt.xlabel('k') 
    # naming the y axis 
    plt.ylabel('accuracy') 
    plt.xticks(k_values)

    plt.plot(k_values, accuracy, '--bo') 
    # giving a title to my graph 
    plt.title('Accuracy over k') 
    
    # function to show the plot 
    if show:
        plt.show() 

    print("Best model with k = "+str(best_k)+" and accuracy "+str(max)+" on validation set")

    if normalize:
        X_test_used = scaler.transform(X_test)
    else:
        X_test_used = X_test

    y_pred_final = best_model.predict(X_test_used)

    print("Accuracy on test set: "+str(metrics.accuracy_score(y_test,y_pred_final)))
    print("--------------------------------") 

def SVM(kernel, normalize, show):
    accuracy=[]

    c_values = [0.001,0.01,0.1,1,10,100,1000]

    max = 0
    scaler = StandardScaler()

    figure, axs = plt.subplots(4,2, figsize=(10,10), constrained_layout=True)
    axs = axs.ravel()
    i = 0
    
    if(normalize):
        scaler.fit(X_train)
        X_train_used = scaler.transform(X_train) # scale train set
        X_val_used = scaler.transform(X_val) # scale validation set with the same parameters
    else:
        X_train_used = X_train
        X_val_used = X_val

    for c in c_values:

        l_svm = svm.SVC(kernel=kernel, C=c, gamma='scale')
        l_svm.fit(X_train_used, y_train)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if show:
            plotDecisionBoundary(X_train_used, y_train, l_svm, c, "c",axs[i])

        y_pred = l_svm.predict(X_val_used)
        acc = metrics.accuracy_score(y_val, y_pred)
        
        if(acc > max):
            best_c = c
            max = acc
            best_model = l_svm
        accuracy.append(acc)
        i+=1

    plt.figure()
    # naming the x axis 
    plt.xlabel('log(c)') 
    # naming the y axis 
    plt.ylabel('accuracy') 
    plt.xticks(c_values)
    plt.xscale('log')

    plt.plot(c_values, accuracy, '--bo') 
    # giving a title to my graph 
    plt.title('Accuracy over k') 
    
    # function to show the plot 
    if show:
        plt.show()
    print("Best model with c = "+str(best_c)+" and accuracy "+str(max)+" on validation set")

    if normalize:
        X_test_used = scaler.transform(X_test)
    else:
        X_test_used = X_test

    y_pred_final = best_model.predict(X_test_used)

    print("Accuracy on test set: "+str(metrics.accuracy_score(y_test,y_pred_final)))
    print("--------------------------------") 

def SVMManualGrid(normalize, show):
    Cs = [1, 10, 100, 1000]
    gammas = [0.001, 0.01, 0.1, 1]

    accuracy = np.zeros((len(Cs), len(gammas)))
    max = 0
    i = j = 0
    scaler = StandardScaler()

    figure, axs = plt.subplots(6,3, figsize=(15,15), constrained_layout=True)
    axs = axs.ravel()
    l = 0

    if(normalize):
        scaler.fit(X_train)
        X_train_used = scaler.transform(X_train) # scale train set
        X_val_used = scaler.transform(X_val) # scale validation set with the same parameters
    else:
        X_train_used = X_train
        X_val_used = X_val

    for c in Cs:
        j = 0
        for gamma in gammas:
            l_svm = svm.SVC(kernel='rbf', C=c, gamma=gamma)
            l_svm.fit(X_train_used, y_train)
            
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if show:
                plotDecisionBoundary(X_train_used, y_train, l_svm, (str(c)+"/"+str(gamma)), "c/gamma", axs[l])
                l+=1

            y_pred = l_svm.predict(X_val_used)
            acc = metrics.accuracy_score(y_val, y_pred)
            
            if(acc > max):
                best_c = c
                best_gamma = gamma
                max = acc
                best_model = l_svm
            accuracy[i][j] = acc
            j+=1
        i+=1

    plt.figure()
    for ind, i in enumerate(Cs):
        plt.plot(gammas, accuracy[ind], '--o', label='C: ' + str(i))
    plt.legend()
    plt.xlabel('Gamma')
    plt.ylabel('Mean score')
    if show:
        plt.show()

    
    print("Best model with c = "+str(best_c)+" and gamma = "+ str(best_gamma) +" and accuracy "+str(max)+" on validation set")

    if normalize:
        X_test_used = scaler.transform(X_test)
    else:
        X_test_used = X_test

    y_pred_final = best_model.predict(X_test_used)

    print("Accuracy on test set: "+str(metrics.accuracy_score(y_test,y_pred_final)))
    print("--------------------------------")   

def SVMGridSearch(normalize, show):
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    gammas = [0.001, 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    scaler = StandardScaler()

    clfs = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=5, iid='False')
    if normalize:
        scaler.fit(X_train_val)
        X_train_used = scaler.transform(X_train_val) # scale train set
    else:
        X_train_used = X_train_val
        
    clfs.fit(X_train_used, y_train_val)

    scores = clfs.cv_results_['mean_test_score']
    scores = np.array(scores).reshape(len(Cs), len(gammas))

    for ind, i in enumerate(Cs):
        plt.plot(gammas, scores[ind], '--o', label='C: ' + str(i))
    plt.legend()
    plt.xlabel('Gamma')
    plt.ylabel('Mean score')
    if show:
        plt.show()

    print("Best model on training set has %r with accuracy %.4f" %(clfs.best_params_, clfs.best_score_))

    if normalize:
        X_test_used = scaler.transform(X_test)
    else:
        X_test_used = X_test

    print("Accuracy on test set: %.4f" %(clfs.score(X_test_used, y_test)))
    print("--------------------------------") 


data, y = load_wine(return_X_y=True)

#print(data.shape)
#print(y.shape)

X = data[:,:2]

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.28, random_state=30)

while(1):
    answer = input("Which classifier you want to use (add norm if you want to normalize data):\n-(1)kNN\n-(2)Linear SVM\n-(3)RBF SVM\n-(4)Manual SVM GridSearch \n-(5)K-Fold SVM GridSearch\n"
                +"-(all)All models\n-(q)Quit the application\nChoise: ")

    if(answer == "1"):
        print("KNN no normalization")
        kNN(False, True)
    if(answer == "1 norm"):
        print("KNN with normalization")
        kNN(True, True)
    if(answer == "2"):
        print("SVM Linear no normalization")
        SVM("linear", False, True)
    if(answer == "2 norm"):
        print("SVM Linear with normalization")
        SVM("linear", True, True)
    if(answer == "3"):
        print("SVM rbf no normalization")
        SVM("rbf", False, True)
    if(answer == "3 norm"):
        print("SVM rbf with normalization")
        SVM("rbf", True, True)
    if(answer == "4"):
        print("Manual Grid no normalization")
        SVMManualGrid(False, True)
    if(answer == "4 norm"):
        print("Manual with normalization")
        SVMManualGrid(True, True)
    if(answer == "5"):
        print("SearchGrid no normalization")
        SVMGridSearch(False, True)
    if(answer == "5 norm"):
        print("SearchGrid with normalization")
        SVMGridSearch(True, True)
    if(answer == "all"):
        print("KNN no normalization")
        kNN(False, False)
        print("KNN with normalization")
        kNN(True, False)
        print("SVM Linear no normalization")
        SVM("linear", False, False)
        print("SVM Linear with normalization")
        SVM("linear", True, False)
        print("SVM rbf no normalization")
        SVM("rbf", False, False)
        print("SVM rbf with normalization")
        SVM("rbf", True, False)
        print("Manual Grid no normalization")
        SVMManualGrid(False, False)
        print("Manual with normalization")
        SVMManualGrid(True, False)
        print("SearchGrid no normalization")
        SVMGridSearch(False, False)
        print("SearchGrid with normalization")
        SVMGridSearch(True, False)
    if(answer=="q"):
        print("Quitting...")
        break
    plt.close(fig="all") 
""" TO DO:
    - comparare i modelli
    - capire come si stampa una griglia
 """