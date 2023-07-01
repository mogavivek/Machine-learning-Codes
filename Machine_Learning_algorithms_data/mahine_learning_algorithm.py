import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class LinearRegrssionAlgorithms():
    
    def __init__(self, X_dataset, y_dataset):
        '''
        This function take the two input parameters as a X and y dataset

        Arguments:
            X_dataset : X columns value from dataset
            y_dataset : y columns value from dataset
        '''
        self.X_dataset = X_dataset
        self.y_dataset = y_dataset

    def logistic_regression(self, test_size, random_state):
        '''
        This function take the dataset input in x and y format\n
        It splits the data set in two method: train and test. Then it predicts the values.\n
        Later it evaluates the score of the prediction
        
        Arguments:
            test_size : to split the dataset in train and test. (provide value in 0.1 to 0.5)
            random_state : depends on the requirement e.g. 1,2,...
        
        Return:
            model : the logistic regression model for further use
            training_data_accuracy : accuracy score of the training data set
            test_data_accuracy : accuracy score of the test data set
        '''
        X_train, X_test, y_train, y_test = train_test_split(self.X_dataset, self.y_dataset, 
                                                test_size=test_size, stratify=self.y_dataset
                                                , random_state=random_state)
        
        #Model training -> Logistic regression
        model = LogisticRegression()
        model.fit(X_train, y_train)

        #Model evaluation for training dataset
        X_train_prdiction = model.predict(X_train)
        training_data_accuracy = accuracy_score(X_train_prdiction, y_train)

        #Model evaluation for training dataset
        X_test_prdiction = model.predict(X_test)
        test_data_accuracy = accuracy_score(X_test_prdiction, y_test)

        return model, training_data_accuracy, test_data_accuracy