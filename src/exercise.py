import csv
from Meals import Meals
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import cross_validation
from sklearn.externals import joblib

import numpy as np
from svm_decision_function import svm_predict_from_decition_function
from __builtin__ import str

    
def predict_meal_category(name, description):
    '''
    Predicts a category based on name and description of a meal.

    Parameters:
        name: name of meal as a unicode string
        description: description of meal as a unicode string

    Returns:
        * sub category, if classifier is sure about it
        * parent category, if classifier is sure about the parent category but not about the sub category
        * None, if classifier thinks it is neither a pizza nor a pasta (or is unsure about both)

        Example returns:

        return 'Pizza Salmone' # if classifier is sure about the sub category

        return 'Pizza' # if classifier is only sure that it is a pizza

        return None # totally unsure if it is a pasta or pizza
    '''
    #make sure name and desc are string
    name = str(name)
    description = str(description)
    
    if not('model_created' in globals()):
        global global_svm_cat
        global global_svm_parent_cat
        global model_created
        global global_decision_fraction
        global count_vect
        global tfidf_transformer
        global meals
        
        global_svm_cat = svm.SVC(probability=True,kernel='linear', C=1)
        global_svm_parent_cat = svm.SVC(probability=True,kernel='linear', C=1)
        model_created = False
    if not(model_created):
        model_created = True
        print "Model is not created yet , creating it"
        meals = Meals()
        with open('../data/meal_descriptions_train.csv', 'rb') as csvfile:
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
        
        global_svm_parent_cat.fit(X_train_tfidf,Y_parent_cat)
        global_svm_cat.fit(X_train_tfidf,Y_cat)
        
        scores_cat = cross_validation.cross_val_score(global_svm_cat,X_train_tfidf, Y_cat, cv=5)
        scores_parent_cat = cross_validation.cross_val_score(global_svm_parent_cat,X_train_tfidf, Y_parent_cat, cv=5)
        print "scores_cat => "+ str(scores_cat) +str(np.mean(scores_cat))
        print "scores_parent_cat =>"+ str(scores_parent_cat) + str(np.mean(scores_parent_cat))
    
    else:
        print "Model already created!"
    #Apply the model
    new_text= name+" "+description
    new_text_list = [new_text]
    X_new_counts = count_vect.transform(new_text_list)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    
    
    num_cats = len(meals.meal_category_value_index)
    num_parent_cats = len(meals.meal_parent_category_value_index)
    
    pred_parent_cat = global_svm_parent_cat.predict_proba(X_new_tfidf)
    pred_parent_cat_index = global_svm_parent_cat.predict(X_new_tfidf)
    pred_parent_cat_max_prob = np.max(pred_parent_cat[0])
    
    pred_cat = global_svm_cat.predict_proba(X_new_tfidf)
    pred_cat_index = global_svm_cat.predict(X_new_tfidf)
    pred_cat_max_prob = np.max(pred_cat[0])
    
    cat_thr = 1.0/num_cats *2 #twice as the random classifier
    parent_cat_thr = 1.0/num_parent_cats*1.5 #1.5 times as random classifier
    
    if(pred_cat_max_prob >=cat_thr):
        pstr = meals.meal_category_index_value.get(pred_cat_index[0])
        return pstr
    if(pred_parent_cat_max_prob >= parent_cat_thr):
        pstr = meals.meal_parent_category_index_value.get(pred_parent_cat_index[0])
        return pstr
    return None
#     
    # TODO: insert call of your classifier code here
    raise NotImplementedError('Please implement the meal category classifier.')
    