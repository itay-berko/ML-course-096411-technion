import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

#adjust_labels_to_binary function get as srguments the training set target y_train as nparry,and a target calss value as string
#it reteruns an nparry with the same shape as y_train with only binary labels: 1 for the target class value and -1 otherwise
def adjust_labels_to_binary(y_train, target_class_value):
    
    target = None 
    
    if target_class_value=='Setosa':
        target = 0

    if target_class_value=='Versicolour':
        target = 1

    if target_class_value=='Virginicacv':
        target = 2
    
    ans = (y == target).astype(int) - (y != target).astype(int)
    return ans.astype(int)
    

#one_vs_rest function gets as arguments x_train and y_train both as nparrays, and target_class_value as string
#it first binarize y_train according to target_class_value via the function adjust_labels_to_binary
#it returns a logistic regression model object trained on x_train and y_binarized     
def one_vs_rest(x_train, y_train, target_class_value):
    y_train_binarized = adjust_labels_to_binary(y_train, target_class_value)
    model = LogisticRegression()
    return model.fit(x_train,y_train_binarized)

    


#binarized_confusiom_matrix gets as arguments X, y_binarized as nparrays, the appropraite one_vs_rest_model as a model object
#and prob_threshold value
#it utilizes one_vs_rest model and predicted probabilities and the prob_threshold to predict
#y_pred on X
#it comparse it to the  y_binarized and 
#return an nparray of the appropriate confusion matrix as follows:
#[TP, FP
#FN, TN]    
def binarized_confusion_matrix(X, y_binarized, one_vs_rest_model, prob_threshold):
        predic = np.array(one_vs_rest_model.predict_proba(X))
    predic_threshold_prob = np.zeros(len(predic))
    
    for i in range(len(predic)):
        if predic[i,1] > prob_threshold:
            predic_threshold_prob[i] = 1
        else:
            predic_threshold_prob[i] = -1
    
    temp = predic_threshold_prob-y_binarized
    tp = sum([1 if predic_threshold_prob[i]==y_binarized[i]==1 else 0 for i in range(len(y_binarized))])
    tn = sum([1 if predic_threshold_prob[i]==y_binarized[i]==-1 else 0 for i in range(len(y_binarized))])
    fp = sum([1 if predic_threshold_prob[i]==1 and y_binarized[i]==-1 else 0 for i in range(len(y_binarized))])
    fn = sum([1 if predic_threshold_prob[i]==-1 and y_binarized[i]==1 else 0 for i in range(len(y_binarized))])
    return np.array([[tp,fn],[fp,tn]])
 
       
    
    
    
   


#micro_avg_precision gets as arguments X, y as nparrays, 
#all_target_class_dict a dictionary with key class value as string with value per key of the approprite one_vs_rest model
#prob_threshold the probability that if greater or equal to the prediction is 1, otherwise -1
#returns the micro average precision
def micro_avg_precision(X, y, all_target_class_dict, prob_threshold):
      tp = 0
    fp = 0
    
    for i in all_target_class_dict:
        y_binarized = adjust_labels_to_binary(y,i)
        c_matrix = binarized_confusion_matrix(X, y_binarized, all_target_class_dict[i], prob_threshold)
        tp = tp + c_matrix[0,0]
        fp = fp + c_matrix[1,0]
        
    return tp/(tp+fp)
    
  


def micro_avg_recall(X, y, all_target_class_dict, prob_threshold):
    tp = 0
    fn = 0
    
    for i in all_target_class_dict:
        y_binarized = adjust_labels_to_binary(y,i)
        c_matrix = binarized_confusion_matrix(X, y_binarized, all_target_class_dict[i], prob_threshold)
        tp = tp + c_matrix[0, 0]
        fn = fn + c_matrix[0, 1]
        
    return tp/(tp+fn)


def micro_avg_false_positve_rate(X, y, all_target_class_dict, prob_threshold):
    fp = 0
    tn = 0
    for i in all_target_class_dict:
        y_binarized = adjust_labels_to_binary(y,i)
        c_matrix = binarized_confusion_matrix(X, y_binarized, all_target_class_dict[i], prob_threshold)
        fp = fp + c_matrix[1, 0]
        tn = tn + c_matrix[1, 1]
    return fp/(tn+fp)
    
   



def f_beta(precision, recall, beta):
    return ((1+beta)^2)*(precision*recall)/(((beta^2)*precision)+recall)
    
  


if __name__ == '__main__':
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=98, test_size=0.3)
