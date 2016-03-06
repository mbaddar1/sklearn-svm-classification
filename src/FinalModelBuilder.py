'''
Created on Feb 24, 2016

@author: baddar
'''

#http://stackoverflow.com/questions/20113206/scikit-learn-svc-decision-function-and-predict
import csv
from Meals import Meals
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import cross_validation
from sklearn.externals import joblib
import numpy as np
from __builtin__ import str

meals = Meals()
with open('../data/meal_descriptions_train_reduced.csv', 'rb') as csvfile:
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

count_vect = CountVectorizer()
X_train_counts =  count_vect.fit_transform(meals.name_and_desc)

tfidf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)

Y_cat = meals.meal_category_mapped
Y_parent_cat = meals.meal_parent_category_mapped


''''
Final SVM model builder
'''
svm_model = svm.SVC(probability=True,kernel='linear', C=1)

svm_model.fit(X_train_tfidf,Y_parent_cat)
# joblib.dump(svm_model, './saved_models/svm_parent_cat.pkl')

# svm_model.fit(X_train_tfidf,Y_cat)
# joblib.dump(svm_model, './saved_models/svm_cat.pkl')
scores_cat = cross_validation.cross_val_score(svm_model,X_train_tfidf, Y_cat, cv=10)
scores_parent_cat = cross_validation.cross_val_score(svm_model,X_train_tfidf, Y_parent_cat, cv=10)
print "scores_cat => "+ str(scores_cat) +str(np.mean(scores_cat))
print "scores_parent_cat =>"+ str(scores_parent_cat) + str(np.mean(scores_parent_cat))
svm_model_cat_loaded = joblib.load('./saved_models/svm_cat.pkl')

test_desc = ["mit Schinken, Paprika, frischen Champignons, Zwiebeln und Oliven"]

X_test_counts = count_vect.transform(test_desc)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
p = svm_model.predict_proba(X_test_tfidf)
print "p=>"+str(p)
print "p shape=>"+str(p.shape)
print "max p =>"+str(np.max(p[0]))
# print p[0]
# print str(type(p))
# print meals.meal_category_index_value.get(p[0])
print "finished"