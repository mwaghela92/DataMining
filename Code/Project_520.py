# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier

####select classifier to run the code
#classifier = 'random_forest'
#classifier = 'svm'
#classifier = 'MLPClassifier'
#classifier = 'gaussian_naive_bayes'
#classifiers = ['gaussian_naive_bayes', 'random_forest', 'svm', 'MLPClassifier' ]
classifiers = ['random_forest']

#### Define directory and data paths
Working_Directory='/Users/mayur/Documents/GitHub/IEE520_Project/DataMining/'
Data_Directory = Working_Directory + 'Raw Data/'

Train = pd.read_csv(Data_Directory + 'Train_Data.csv').dropna()
Test = pd.read_csv(Data_Directory + 'Test_Data.csv')


def ONE_HOT_ENCODING(dataframe, column, name):
    """
    This function does one-hot-encoding for
    data frame = dataframe
    column of data frame (column converted to dataframe) = column
    name of the column = name
    
    returns:
        data frame with onehotencoding of mentioned column 
        (original column is deleted)
    """
    
    
    le = LabelEncoder()
    labels = column.apply(le.fit_transform)
    enc = OneHotEncoder()
    enc.fit(labels)
    onehotlabels = enc.transform(labels).toarray()
    dataframe = dataframe.join(pd.DataFrame(onehotlabels), lsuffix='_left', 
                               rsuffix='_right')
    dataframe = dataframe.drop(name, axis = 1)
    return dataframe


def ACCURACY_EVAL(y_true, y_pred, classifier):
    """ This function claculates the total error rate, 
    balanced error rate and plots a confusion matrix"""
    
    ter = (100 - accuracy_score(y_true, y_pred) *100)
    print('Total error rate:', ter)
    print(confusion_matrix(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    mean_acc=((fp/(tn+fp)) + (fn/(fn+tp)))/2
    print('Balanced error rate', (mean_acc *100))
    df_cm = pd.DataFrame(cm)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    Title = (classifier + '\nBalanced Error Rate:' + str(mean_acc *100) +
              '\nTotal error rate: ' + str(ter))
    plt.title(Title)
    my_plot = plt.gcf()
    plt.savefig(Working_Directory + 'Results/' + classifier +'_cm.png')
    
def CLASSIFIER_SELECTION(classifier):
    """This function returns the initialization and parameter 
        dictionary of selected 'classifier'"""
        
    if classifier == 'random_forest':
        c = RandomForestClassifier(verbose = True, random_state = 235)
        parameters = {'n_estimators': [50, 100, 200], 
              'max_depth': [2,5,10, 30, None],
              'max_features': ['auto', 'sqrt', 'log2', None]
                      }
    if classifier == 'svm':
        c = svm.SVC(verbose = True)
        """
        parameters = {'C' : [0.01, 0.1, 1, 10, 100],
                      'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                      'gamma': ['auto', 'scale'],
                      'cache_size' : [10, 100],
                      }"""
        parameters = {'C' : [0.1, 1],
                      'kernel': ['rbf'],
                      }
    if classifier == 'MLPClassifier':
        c = MLPClassifier(learning_rate = 'adaptive', verbose = True)
        parameters = {'hidden_layer_sizes':[(100,), (100,100)],
                       'solver':['lbfgs','sgd', 'adam'],
                       'alpha': [0.001, 0.0001, 0.1]
                       }
    if classifier == 'gaussian_naive_bayes':
        c = GaussianNB()
        parameters = {}
        
    return (c,parameters)
    
    
    
Encoding_list = ['x5', 'x13', 'x64', 'x65']
for i in Encoding_list:
    Train = ONE_HOT_ENCODING(Train,pd.DataFrame(Train[i]), i)
#for i in Encoding_list:
 #   Test = ONE_HOT_ENCODING(Test,pd.DataFrame(Test[i]), i)
    

    
##### checking if there exists a class imbalance
print('Train class balance check:\n',Train['y'].value_counts())


##### resample minority class
df_minority = resample(Train[Train.y==1], 
                                 replace=True,     
                                 n_samples=1282,    
                                 random_state=123)

New_Train = pd.concat([Train, df_minority])
print('New_Train class balance check:\n',New_Train['y'].value_counts())



##### separate target from the dataframe and splitting train and test
y = New_Train['y']
X = New_Train.drop('y', axis = 1)
scaler = MinMaxScaler()
X_scaled = scaler.fit(X).transform(X)
"""
y = Train['y']
X = Train.drop('y', axis = 1)
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                test_size=0.20, random_state=42)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_scaled, y, 
                                test_size=0.20, random_state=42)

for i in classifiers:
    """
    if i not in ['MLPClassifier']:
        c, parameters = CLASSIFIER_SELECTION(i)
    
        gs = GridSearchCV(c, parameters, cv=5)
        #clf = gs.best_params_
        gs.fit(X_train, y_train)
        print(gs.best_params_)
        y_pred = gs.predict(X_test)
    
        ACCURACY_EVAL(y_test, y_pred, i)
    else:"""
    c, parameters = CLASSIFIER_SELECTION(i)
    
    clf = GridSearchCV(c, parameters, cv=5)
    
    clf.fit(X_train_s, y_train_s)
    y_pred = clf.predict(X_test_s)
    
    ACCURACY_EVAL(y_test_s, y_pred, i)

    #clf.get_params()
"""
from sklearn.linear_model import SGDClassifier
c = RandomForestClassifier()
#c = svm.SVC()
ad = AdaBoostClassifier(c)
ad.fit(X_train_s, y_train_s)
y_pred = ad.predict(X_test_s)
ACCURACY_EVAL(y_test_s, y_pred, 'ada')
""""







