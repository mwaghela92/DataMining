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
import numpy as np

####select classifier to run the code
classifiers = ['gaussian_naive_bayes', 'random_forest', 'svm', 
               'MLPClassifier', 'adaboost' ]
 #classifiers = ['random_forest']

#### Define directory and data paths
Working_Directory='/Users/mayur/Documents/GitHub/IEE520_Project/DataMining/'
Data_Directory = Working_Directory + 'Raw Data/'

Train = pd.read_csv(Data_Directory + 'Train_Data.csv').dropna()
Test = pd.read_csv(Data_Directory + 'Test_Data.csv')
Train = Train.drop('Row', axis = 1)
Test = Test.drop('Row', axis = 1)


def ONE_HOT_ENCODING(dataframe, column, name, df, cl):
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
    labels_test = cl.apply(le.transform)
    enc = OneHotEncoder()
    enc.fit(labels)
    onehotlabels = enc.transform(labels).toarray()
    ohl_test = enc.transform(labels_test).toarray()
    dataframe = dataframe.join(pd.DataFrame(onehotlabels), lsuffix='_left', 
                               rsuffix='_right')
    df = df.join(pd.DataFrame(ohl_test), lsuffix='_left', 
                               rsuffix='_right')
    dataframe = dataframe.drop(name, axis = 1)
    df = df.drop(name, axis = 1)
    return dataframe, df


def ACCURACY_EVAL(y_true, y_pred, classifier):
    """ This function claculates the total error rate, 
    balanced error rate and plots a confusion matrix"""
    
    ter = (100 - accuracy_score(y_true, y_pred) *100)
    print('Total error rate:', ter)
    print(confusion_matrix(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    mean_acc=((fp/(tn+fp)) + (fn/(fn+tp))) *100/2
    print('Balanced error rate', (mean_acc))
    df_cm = pd.DataFrame(cm)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    Title = (classifier + '\nBalanced Error Rate:' + str(mean_acc) +
              '\nTotal error rate: ' + str(ter))
    plt.title(Title)
    my_plot = plt.gcf()
    plt.savefig(Working_Directory + 'Results/' + classifier +'_cm.png')
    return (ter,mean_acc)
    
def CLASSIFIER_SELECTION(classifier):
    """This function returns the initialization and parameter 
        dictionary of selected 'classifier'"""
        
    if classifier == 'random_forest':
        c = RandomForestClassifier(verbose = True, random_state = 17, 
                                   class_weight='balanced', criterion = 'entropy')
        
        parameters = {'n_estimators': [10, 100, 400], 
              'max_depth': [10, 30, None]
                      }
        """
        parameters = {'max_depth': [10],  
                         'n_estimators': [10,100]}"""
        
    if classifier == 'svm':
        c = svm.SVC(verbose = True,  random_state = 235, probability = True,
                    class_weight='balanced')
        parameters = {'C' : [0.1, 10, 1000],
                      'kernel': ['rbf'],
                      'gamma' : [0.001,1],
                      }
    if classifier == 'MLPClassifier':
        c = MLPClassifier(learning_rate = 'adaptive', verbose = True, 
                          random_state = 23, alpha = 0.1,
                          )
        parameters = {'hidden_layer_sizes':[(100,), (100,100)],
                       'solver':['sgd', 'adam'],
                       'activation': ['logistic'],
                       'alpha': [0.001, 0.0001, 0.1]
                       }
    if classifier == 'gaussian_naive_bayes':
        c = GaussianNB()
        parameters = {
                        }
    if classifier == 'adaboost':
        rf= RandomForestClassifier(random_state = 235, class_weight='balanced')
        gnb = GaussianNB()
        c = AdaBoostClassifier()
        parameters = { 'base_estimator': [ rf, gnb],
                      'n_estimators' : [10,50,100]}
        
        
    return (c,parameters)
    
    
    
Encoding_list = ['x5', 'x13', 'x64', 'x65']
for i in Encoding_list:
    Train, Test = ONE_HOT_ENCODING(Train,pd.DataFrame(Train[i]), i,
                             Test, pd.DataFrame(Test[i]))
#for i in Encoding_list:
 #   Test = ONE_HOT_ENCODING(Test,pd.DataFrame(Test[i]), i)
    
y = Train['y']
X_train, X_test, y_train, y_test = train_test_split(Train, y, 
                                test_size=0.20, random_state=42, shuffle=True)


##### checking if there exists a class imbalance
print('Train class balance check:\n',X_train['y'].value_counts())


##### resample minority class
df_minority = resample(X_train[X_train.y==1], 
                                 replace=True,     
                                 n_samples=1000,    
                                 random_state=12)
df_majority = resample(X_train[X_train.y==-1], 
                                 replace=True,     
                                 n_samples=1000,    
                                 random_state=12)

New_Train = pd.concat([df_majority, df_minority])

print('New_Train class balance check:\n',New_Train['y'].value_counts())



##### separate target from the dataframe and splitting train and test
y_train = X_train['y']
X_train = X_train.drop('y', axis = 1)
X_test = X_test.drop('y', axis = 1)
scaler = MinMaxScaler()
X_train_s = scaler.fit(X_train).transform(X_train)
X_test_s = scaler.transform(X_test)
Test = scaler.transform(Test)
y_train_s = y_train
y_test_s = y_test
"""
y = Train['y']
X = Train.drop('y', axis = 1)
"""


best_parameters = {}
accuracy_scores = {}
for i in classifiers:
    
    if i not in ['MLPClassifier', 'adaboost', 'svm']:
        ac_scores = {}
        c, parameters = CLASSIFIER_SELECTION(i)
    
        clf = GridSearchCV(c, parameters, refit = True,cv=5)
        
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        
        ter, ber = ACCURACY_EVAL(y_test, y_pred, i)
        ac_scores['Total Error Rate'] = ter
        ac_scores['Balanced Error Rate'] = ber
        ac_scores['Mean CV error'] = (1-clf.best_score_ ) *100
        accuracy_scores[i] = ac_scores
        
        best_parameters[i] = clf.best_params_
    else:
        ac_scores = {}
        c, parameters = CLASSIFIER_SELECTION(i)
        
        clf = GridSearchCV(c,  parameters, cv=5, refit = True)
        
        clf.fit(X_train_s, y_train_s)
        
        y_pred = clf.predict(X_test_s)
        
        ter, ber = ACCURACY_EVAL(y_test_s, y_pred, i)
        ac_scores['Total Error Rate'] = ter
        ac_scores['Balanced Error Rate'] = ber
        ac_scores['Mean CV error'] = (1-clf.best_score_ )*100
        accuracy_scores[i] = ac_scores
        
        best_parameters[i] = clf.best_params_


### Using the best classifier to predict the values of the given test data
best_clf = svm.SVC(C = 1000, gamma = 0.001, kernel = 'rbf', verbose = True,  
                   random_state = 235, probability = True,
                    class_weight='balanced')

best_clf.fit(X_train_s, y_train_s)
#y_pred = best_clf.predict(X_test_s)
#ACCURACY_EVAL(y_test_s, y_pred, '1')
test_prediction = best_clf.predict(Test)
row_number = np.linspace(1,1647,1647)

file_name = 'BMI555IEE520_Results2018_MayurWaghela.csv'
final_result = pd.DataFrame(row_number)
final_result['pred'] = test_prediction
final_result.to_csv(Working_Directory + 'Results/' + file_name, 
                    header=None, index = False )

print(accuracy_scores)

