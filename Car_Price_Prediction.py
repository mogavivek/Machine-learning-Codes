import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
import Machine_Learning_algorithms_data.data_information as data_info
import Machine_Learning_algorithms_data.mahine_learning_algorithm as ml_algorithm
from typing import Self

#Now loading the data from csv file
file_path = "C:\\Users\\vivek\\PycharmProjects\\pythonProject\\Vivekcode\\Machine-learning-Codes\\data-set\\car data.csv"
car_dataset_file_path = data_info.DataDescription(file_path)
car_dataset, car_dataset_head, car_dataset_description, car_dataset_info, car_dataset_null_values = car_dataset_file_path.data_description()
print(car_dataset_head)
print(car_dataset_description)
print(car_dataset_info)
print(car_dataset_null_values)

#Checing the distribution of categorical data
print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Seller_Type.value_counts())
print(car_dataset.Transmission.value_counts())

#encoding the categorical data
car_dataset.replace({'Fuel_Type': {'Petrol':0, 'Diesel':1, 'CNG':2}}, inplace=True)
car_dataset.replace({'Transmission': {'Manual':0, 'Automatic':1}}, inplace=True)
car_dataset.replace({'Seller_Type': {'Dealer':0, 'Individual':1}}, inplace=True)

print(car_dataset.head())

#Splitting data into training data and test data
X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
y = car_dataset.iloc[:,2]

lasso_model = ml_algorithm.LinearRegrssionAlgorithms(X, y)
model, error_score, train_data_accuracy, test_data_accuracy = lasso_model.lasso_regression(0.1, 2)

print(error_score)
print(train_data_accuracy)
print(test_data_accuracy)
