'''
Created on Feb 25, 2016

@author: baddar
'''


    
from sklearn import svm
import numpy as np
from __builtin__ import str
#ref
#http://stackoverflow.com/questions/20113206/scikit-learn-svc-decision-function-and-predict
def svm_predict_from_decition_function(svm_decision_function_result,nclasses):
    tmp = svm_decision_function_result
    if not(isinstance(tmp,(tuple,np.ndarray,list))):
        l = list()
        l.append(tmp)
        svm_decision_function_result = l
        
    votes = np.zeros(nclasses)
    p = 0
    for i in range(1,nclasses+1,1):
        for j in range(i+1,nclasses+1,1): #compare each class i to class j where j > i 1<=i,j<=nclasses
            if svm_decision_function_result[p] > 0:
                votes[i-1] += 1
            else: 
                votes[j-1] += 1
            p += 1
    
    max_index = np.argmax(votes)
    sorted_votes = np.sort(votes)
    n = len(sorted_votes)
    diff = sorted_votes[n-1] - sorted_votes[n-2]
    if(diff <=0):
        return None
    else:
        return (max_index+1)
    
# y = [1,1,2,2,3,3,4,4]
# X = np.random.randn(8, 10)
# svm = svm.SVC().fit(X,y)
# for sample_index in range(0,8,1):
#     result = svm.decision_function(X)[sample_index]
#     decision_class = svm_predict_from_decition_function(result, 4)
#     decision_class_2 = svm.predict(X)[sample_index]
#     print (decision_class,decision_class_2)
# # print result.shape
# # print result