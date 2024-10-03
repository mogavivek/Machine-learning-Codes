import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataDescription():

    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
    
    def data_description(self):
        '''
        This function take the input of the csv file and then read the required data

        Return:
            df = dataset\n
            dataset_head = head values of the table\n
            dataset_describe = description of the dataset\n
            dataset_info = info of all columns as well as null values\n
            dataset_null_values = total null values in that column\n
        '''
        df = pd.read_csv(self.csv_file_path)
        dataset_head = df.head()
        dataset_describe = df.describe()
        dataset_info = df.info()
        dataset_null_values = df.isnull().sum()

        return df, dataset_head, dataset_describe, dataset_info, dataset_null_values
    
    def plot_classification_in_seaborn(dataset, column_name_1, column_name_2):
        """
        This function plot the two values from the dataset to do classification\n
        Then it will also plot the graph hence no need to put extra plt.show() in code
        Arguments:
            dataset : dataset in csv format
            column_name_1: choose the column from dataset
            column_name_2: choose the second column from the dataset for comarision

        Return:
            changed column with numerical value
        """
        sns.countplot(x='{}'.format(column_name_1), hue='{}'.format(column_name_2), data=dataset)
        plt.show()
    
    def fill_na_with_mean_values_of_selected_column(self, dataset:tuple, column_name:str):
        """This function fill the null values with the mean values\n
        Arguments:
              dataset : dataset in csv format
              column_name : It takes the string value, The name of the column which want to replace null values with mean      
        """
        df = pd.read_csv(self.csv_file_path)
        df[column_name].fillna(df[column_name].mean(), inplace=True)




