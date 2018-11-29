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


def ACCURACY_EVAL(y_true, y_pred):
    print(accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(tn, fp)
    print(fn, tp)
    
    
    
    
    
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



##### separate from the dataframe and splitting train and test
y = New_Train['y']
X = New_Train.drop('y', axis = 1)
"""
y = Train['y']
X = Train.drop('y', axis = 1)
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                test_size=0.33, random_state=42)


clf = RandomForestClassifier(n_estimators=1000, max_depth=50,
                              random_state=10)
clf = svm.SVC()

clf = MLPClassifier(hidden_layer_sizes=(500, 500, 500), solver='sgd', 
                    verbose= True, max_iter=2000, learning_rate_init=0.00000001 , 
                    learning_rate = 'adaptive')

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

ACCURACY_EVAL(y_test, y_pred)
















