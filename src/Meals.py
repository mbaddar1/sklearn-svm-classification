'''
Created on Feb 22, 2016

@author: baddar
'''
from twisted.python.util import println

class Meals(object):
    '''
    classdocs
    '''


    def __init__(self, params=None):
        '''
        Constructor
        '''
        self.id = list()
        self.name = list()
        self.name_and_desc = list()
        self.description = list()
        self.meal_category = list()
        self.meal_parent_category = list()
        self.meal_category_value_index = dict()
        self.meal_category_index_value = dict()
        self.meal_category_mapped = list()
        self.meal_parent_category_value_index = dict()
        self.meal_parent_category_index_value = dict()
        self.meal_parent_category_mapped = list()
        
    def disp(self):
        println(self.id)
        println(self.name)
        println(self.description)
        println(self.meal_category)
        println(self.meal_parent_category)
    
    def calcMealCategoryIndex(self):
        distict_vals = set(self.meal_parent_category)
        idx = 1
        for val in distict_vals:
            self.meal_parent_category_value_index[val] = idx
            self.meal_parent_category_index_value[idx] = val
            idx = idx+1
        for val in self.meal_parent_category:
            self.meal_parent_category_mapped.append(self.meal_parent_category_value_index[val])
        
        distict_vals = set(self.meal_category)
        idx = 1
        for val in distict_vals:
            self.meal_category_value_index[val] = idx
            self.meal_category_index_value[idx] = val
            idx = idx+1
        for val in self.meal_category:
            self.meal_category_mapped.append(self.meal_category_value_index[val])
#         print len(distict_vals)
#         print len(self.meal_category_value_index)
#         print len(self.meal_category_mapped)
#         print self.meal_category_value_index
#         print self.meal_category_mapped
    def printStat(self):
        print "Number of categories = "+str(len(self.meal_category_value_index))
        print "Number of parent categoris = "+str(len(self.meal_parent_category_value_index))
        