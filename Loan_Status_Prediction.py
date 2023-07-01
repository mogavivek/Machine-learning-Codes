import pandas as pd
import numpy as np
import Machine_Learning_algorithms_data.data_information as data_info
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

csv_path = 'C:/Users/vivek/PycharmProjects/pythonProject/Vivekcode/Machine-learning-Codes/data-set/train_u6lujuX_CVtuZ9i (1).csv'

loan_dataset_df = data_info.DataDescription(csv_path)
loan_dataset, loan_dataset_head, loan_dataset_describe, loan_dataset_info, loan_dataset_null_values = loan_dataset_df.data_description()

print(loan_dataset_head)
print(loan_dataset_null_values)

#Now dropping the missing values
loan_dataset = loan_dataset.dropna()
print(loan_dataset.isnull().sum())

#Now replacing the loan status in numerical value by using label encoding
loan_dataset.replace({"Loan_Status":{"N":0, "Y":1}}, inplace=True)
print(loan_dataset.head())

#Counting the dependent total values
print(loan_dataset['Dependents'].value_counts())

#Now replacing the value of 3+ in 4 in Dependents column
loan_dataset = loan_dataset.replace(to_replace="3+", value=4)
print(loan_dataset['Dependents'].value_counts())

"""Data Visulization"""
#educaiton and loan status
sns.countplot(x='Education', hue='Loan_Status', data=loan_dataset)

