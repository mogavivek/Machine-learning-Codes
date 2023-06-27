import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


#Data collection and data processing
sonar_data = pd.read_csv("C:/Users/vivek/PycharmProjects/pythonProject/Vivekcode/Machine-learning-Codes/data-set/Copy of sonar data.csv")
print(sonar_data.head())
sonar_data.shape
sonar_data.describe()
#60 shows the column of M and R
print(sonar_data.iloc[:,60].value_counts())

X = sonar_data.iloc[:, :60]
print(X.shape)
y = sonar_data.iloc[:,60]
print(y.shape)
