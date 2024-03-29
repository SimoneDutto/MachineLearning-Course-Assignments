from sklearn import neighbors, svm, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import util as ut
import numpy as np
import pandas as pd


def kNN(normalize, show):
    accuracy=[]

    k_values = [1,3,5,7]

    max = 0
    
    if(normalize):
        X_train_used = X_train_scaled # scale train set
        X_val_used = X_val_scaled # scale validation set with the same parameters
        X_test_used = X_test_scaled
        X_train_val_used = X_train_val_scaled
    else:
        X_train_used = X_train
        X_val_used = X_val
        X_test_used = X_test
        X_train_val_used = X_train_val
    
    figure, axs = plt.subplots(2,2, figsize=(20,25), constrained_layout=True)
    axs = axs.ravel()
    i = 0

    for n_neighbors in k_values:

        knn = neighbors.KNeighborsClassifier(n_neighbors)
        knn.fit(X_train_used, y_train)
        
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if show:
            ut.plotDecisionBoundary(X_train_used, y_train, knn, n_neighbors, "k", axs[i])

        y_pred = knn.predict(X_val_used)
        acc = metrics.accuracy_score(y_val, y_pred)
        
        if(acc > max):
            best_k = n_neighbors
            max = acc
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

    knn_final = neighbors.KNeighborsClassifier(best_k)
    knn_final.fit(X_train_val_used, y_train_val)

    print("Best model with k = "+str(best_k)+" and accuracy "+str(max)+" on validation set")        

    y_pred_final = knn_final.predict(X_test_used)

    print("Accuracy on test set: "+str(metrics.accuracy_score(y_test,y_pred_final)))
    print("--------------------------------") 

def SVM(kernel, normalize, show):
    accuracy=[]

    c_values = [0.001,0.01,0.1,1,10,100,1000]

    max = 0

    figure, axs = plt.subplots(4,2, figsize=(10,10), constrained_layout=True)
    axs = axs.ravel()
    i = 0
    
    if(normalize):
        X_train_used = X_train_scaled # scale train set
        X_val_used = X_val_scaled # scale validation set with the same parameters
        X_test_used = X_test_scaled
        X_train_val_used = X_train_val_scaled
    else:
        X_train_used = X_train
        X_val_used = X_val
        X_test_used = X_test
        X_train_val_used = X_train_val

    for c in c_values:

        l_svm = svm.SVC(kernel=kernel, C=c, gamma='scale')
        l_svm.fit(X_train_used, y_train)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if show:
            ut.plotDecisionBoundary(X_train_used, y_train, l_svm, c, "c",axs[i])

        y_pred = l_svm.predict(X_val_used)
        acc = metrics.accuracy_score(y_val, y_pred)
        
        if(acc > max):
            best_c = c
            max = acc
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
    final_l_svm = svm.SVC(kernel=kernel, C=best_c, gamma='scale')
    final_l_svm.fit(X_train_val_used, y_train_val)
    y_pred_final = final_l_svm.predict(X_test_used)

    print("Accuracy on test set: "+str(metrics.accuracy_score(y_test,y_pred_final)))
    print("--------------------------------") 

def SVMManualGrid(normalize, show):
    Cs = [0.1, 1, 10, 100, 1000, 10000]
    gammas = [0.001, 0.01, 0.1, 1, 10, 100]

    accuracy = np.zeros((len(Cs), len(gammas)))
    max = 0
    i = j = 0
    scaler = StandardScaler()

    #figure, axs = plt.subplots(len(Cs),len(gammas), figsize=(15,15), constrained_layout=True)
    #axs = axs.ravel()
    l = 0

    if(normalize):
        X_train_used = X_train_scaled # scale train set
        X_val_used = X_val_scaled # scale validation set with the same parameters
        X_test_used = X_test_scaled
        X_train_val_used = X_train_val_scaled
    else:
        X_train_used = X_train
        X_val_used = X_val
        X_test_used = X_test
        X_train_val_used = X_train_val

    for c in Cs:
        j = 0
        for gamma in gammas:
            l_svm = svm.SVC(kernel='rbf', C=c, gamma=gamma)
            l_svm.fit(X_train_used, y_train)
            
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if show:
                #plotDecisionBoundary(X_train_used, y_train, l_svm, (str(c)+"/"+str(gamma)), "c/gamma", axs[l])
                l+=1

            y_pred = l_svm.predict(X_val_used)
            acc = metrics.accuracy_score(y_val, y_pred)
            
            if(acc > max):
                best_c = c
                best_gamma = gamma
                max = acc
            accuracy[i][j] = acc
            j+=1
        i+=1

  
    if show:
        im, cbar = ut.heatmap(accuracy, Cs, gammas, cmap="RdYlGn")
        ut.annotate_heatmap(im)
        plt.show()

    
    print("Best model with c = "+str(best_c)+" and gamma = "+ str(best_gamma) +" and accuracy "+str(max)+" on validation set")

    final_rbf_svm = svm.SVC(kernel='rbf', C=best_c, gamma=best_gamma)
    final_rbf_svm.fit(X_train_val_used, y_train_val)
    y_pred_final = final_rbf_svm.predict(X_test_used)

    print("Accuracy on test set: "+str(metrics.accuracy_score(y_test,y_pred_final)))
    print("--------------------------------")   

def SVMGridSearch(normalize, show):
    Cs = [0.01, 0.1, 1, 10, 100, 1000]
    gammas = [0.001, 0.01, 0.1, 1, 10, 100]
    param_grid = {'svm__C': Cs, 'svm__gamma' : gammas}


    if normalize:
        pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', svm.SVC(kernel='rbf'))])
    else:
         pipe = Pipeline([
              ('svm', svm.SVC(kernel='rbf'))])

    
    clfs = GridSearchCV(pipe, param_grid, cv=5, iid='False')
        
    clfs.fit(X_train_val, y_train_val)

    scores = clfs.cv_results_['mean_test_score']
    scores = np.array(scores).reshape(len(Cs), len(gammas))

    if show:
        im, cbar = ut.heatmap(scores, Cs, gammas, cmap="RdYlGn")
        ut.annotate_heatmap(im)
        plt.show()

    print("Best model on training set has %r with accuracy %.4f" %(clfs.best_params_, clfs.best_score_))
        

    print("Accuracy on test set: %.4f" %(clfs.score(X_test, y_test)))
    print("--------------------------------") 

def analizeData():
    datas = pd.DataFrame(data)
    corr = datas.corr()
    #print(corr.shape)
    plt.figure(figsize=(10,10))
    

    im, cbar= ut.heatmap(corr,np.arange(corr.shape[1]), np.arange(corr.shape[1]), cmap="RdYlGn")
    texts = ut.annotate_heatmap(im)
    
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y,
                edgecolor='k', s=20)
    plt.xlabel("Alcohol")
    plt.ylabel("Malic acid")
    plt.title("Wine classification dataset", pad=10)
    plt.show()


data, y = load_wine(return_X_y=True)

#print(data.shape)
#print(y.shape)

# change this to change attributes
attribute1 = 0
attribute2 = 11

X = data[:,[attribute1, attribute2]]
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=1) 
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=2/7, random_state=30) 

# normalized data
scaler = StandardScaler()
scaler.fit(X_train_val)
X_train_scaled = scaler.transform(X_train)
X_val_scaled =  scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# normalized data for the cross-validation
scaler1 = StandardScaler()
scaler1.fit(X_train_val)
X_train_val_scaled = scaler1.transform(X_train_val)
X_test_scaleds = scaler1.transform(X_test)


#print(X_train.shape)
#print(X_val.shape)
#print(X_test.shape)

while(1):
    answer = input("Which classifier you want to use (add norm if you want to normalize data):\n-(1)kNN\n-(2)Linear SVM\n-(3)RBF SVM\n-(4)Manual SVM GridSearch \n-(5)K-Fold SVM GridSearch\n"
                +"-(all)All models\n-(datainf) For data statistics\n-(q)Quit the application\nChoise: ")

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
    if(answer=="datainf"):
        analizeData()
    if(answer=="q"):
        print("Quitting...")
        break
    plt.close(fig="all") 
""" TO DO:
    - comparare i modelli
    - capire come si stampa una griglia
 """
