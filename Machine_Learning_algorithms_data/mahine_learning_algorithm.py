import numpy as np
import pandas as pd
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn import svm
from sklearn import metrics

class PredictionMethod(Enum):
    Regression = "regression problem"
    Classification = "classification problem" 

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
    
    def support_vector_machine_algorithem(self, expected_problem, test_size, random_state):
        '''
        This function take the dataset input in x and y format\n
        It splits the data set in two method: train and test. Then it predicts the values.\n
        Later it evaluates the score of the prediction
        
        Arguments:
            expected_problem : please choose the method e.g., regression problem or classification problem
            test_size : to split the dataset in train and test. (provide value in 0.1 to 0.5)
            random_state : depends on the requirement e.g. 1,2,...
        
        Return:
            model : the svm model for further use
            training_data_accuracy : accuracy score of the training data set
            test_data_accuracy : accuracy score of the test data set
        '''
        X_train, X_test, y_train, y_test = train_test_split(self.X_dataset, self.y_dataset, 
                                                test_size=test_size, stratify=self.y_dataset
                                                , random_state=random_state)
        
        if(expected_problem == PredictionMethod.Regression.value):
            pass
        elif(expected_problem == PredictionMethod.Classification.value):
            #Training the model (This is classification problem hence SVC used)
            classifier = svm.SVC(kernel="linear")

            #Training the support vectore machine problem
            classifier.fit(X_train, y_train)

            #Model evaluation on accuracy score - traininng
            X_train_preiction = classifier.predict(X_train)
            training_data_accuracy = accuracy_score(X_train_preiction, y_train)

            #Model evaluation on accuracy score - test
            X_test_prediciton = classifier.predict(X_test)
            test_data_accuracy = accuracy_score(X_test_prediciton, y_test)

            return classifier, training_data_accuracy, test_data_accuracy
    
    def lasso_regression(self, test_size, random_state):
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
                                                test_size=test_size, random_state=random_state)
        
        #Model training -> Logistic regression
        model = Lasso()
        model.fit(X_train, y_train)

        #Model error score for training dataset
        X_train_prdiction = model.predict(X_train)
        error_score = metrics.r2_score(y_train, X_train_prdiction)

        #Model evaluation on accuracy score - traininng
        X_train_preiction = model.predict(X_train)
        training_data_accuracy = accuracy_score(X_train_preiction, y_train)

        #Model evaluation for training dataset
        X_test_prdiction = model.predict(X_test)
        test_data_accuracy = accuracy_score(X_test_prdiction, y_test)

        return model, error_score, training_data_accuracy, test_data_accuracy

