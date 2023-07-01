import pandas as pd
import numpy as np

class DataDescription():

    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
    
    def data_description(self):
        '''
        This function take the input of the csv file and then read the required data

        Return:
            df = dataset
            dataset_head = head values of the table
            dataset_describe = description of the dataset
            dataset_info = info of all columns as well as null values
            dataset_null_values = total null values in that column
        '''
        df = pd.read_csv(self.csv_file_path)
        dataset_head = df.head()
        dataset_describe = df.describe()
        dataset_info = df.info()
        dataset_null_values = df.isnull().sum()

        return df, dataset_head, dataset_describe, dataset_info, dataset_null_values