'''
Created on Feb 23, 2016

@author: baddar
'''
#http://stackoverflow.com/questions/20113206/scikit-learn-svc-decision-function-and-predict

import csv
import Meals
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import cross_validation
import numpy as np
meals = Meals()
with open('../../data/meal_descriptions_train.csv', 'rb') as csvfile:
    r = csv.reader(csvfile, delimiter=',', quotechar='"')
    rowIndex=0
    for row in r:
        if rowIndex >0:
            meals.id.append(row[0])
            meals.name.append(row[1])
            meals.description.append(row[2])
            meals.name_and_desc.append(row[1]+" "+row[2])
            meals.meal_category.append(row[3])
            meals.meal_parent_category.append(row[4])
        rowIndex  = rowIndex +1
meals.calcMealCategoryIndex()
meals.printStat()
count_vect = CountVectorizer()
X_train_counts =  count_vect.fit_transform(meals.description)

tfidf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)

Y_cat = meals.meal_category_mapped
Y_parent_cat = meals.meal_parent_category_mapped



''''
SVM experiment
'''
#, 'poly','rbf', 'sigmoid'
#np.arange(0.1,1.1,0.1)
kernels = ['linear','poly', 'rbf', 'sigmoid']
C_vals = [0.1,1,10]
# for i in C_vals:
#     C_vals.append(i)
svm_cat_results = list()
svm_parent_cat_results = list()
for C in C_vals:
    for kernel in kernels:
        svm_model_desc = "SVM Model : kernel = "+kernel+" C = "+str(C)
        print "Building "+svm_model_desc
        svm_model = svm.SVC(kernel=kernel, C=C)
        
        scores_cat = cross_validation.cross_val_score(svm_model,X_train_tfidf, Y_cat, cv=5)
        scores_parent_cat = cross_validation.cross_val_score(svm_model,X_train_tfidf, Y_parent_cat, cv=5)
        svm_cat_results.append((svm_model_desc,scores_cat,np.mean(scores_cat)))
        svm_parent_cat_results.append((svm_model_desc,scores_parent_cat,np.mean(scores_parent_cat)))
print "SVM Cat results"
print svm_cat_results
print "SVM Parent Cat Results"
print svm_parent_cat_results

max_accuracy = 0
idx=0
max_accuracy_index = -1
for res in svm_cat_results:
    if res[2] > max_accuracy:
        max_accuracy_index = idx
        max_accuracy = res[2]
print "For cat classification"
print "max avg accuracy = "+str(max_accuracy)
print "Best model is "+svm_cat_results[max_accuracy_index][0]

max_accuracy = 0
idx=0
max_accuracy_index = -1
for res in svm_parent_cat_results:
    if res[2] > max_accuracy:
        max_accuracy_index = idx
        max_accuracy = res[2]
print "For parent cat classification"
print "max avg accuracy = "+str(max_accuracy)
print "Best model is "+svm_parent_cat_results[max_accuracy_index][0]
