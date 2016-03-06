'''
Created on Feb 25, 2016

@author: baddar
'''
import exercise
from exercise import predict_meal_category

new_name_1 = "Pizza Capricciosa"
new_desc_1 = "mit Schinken, Paprika, frischen Champignons, Zwiebeln und Oliven"

new_name_2 = "Gnocchi Gorgonzola"
new_desc_2 = "mit Gorgonzola"
print "Hello"
p1 = predict_meal_category(name=new_name_1, description=new_desc_1)
p2 = predict_meal_category(name=new_name_2, description=new_desc_2)
print(p1,p2)