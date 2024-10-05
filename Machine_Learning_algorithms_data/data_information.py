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
            column_name_2: choose the second column from the dataset for comparison

        Return:
            changed column with numerical value
        """
        sns.countplot(x='{}'.format(column_name_1), hue='{}'.format(column_name_2), data=dataset)
        plt.show()
    
    def fill_na_with_mean_values_of_selected_column(self, dataset:tuple, column_name:str):
        """This function fill the null values with the mean values\n
        Arguments:\n
              dataset : dataset in csv format
              column_name : It takes the string value, The name of the column which want to replace null values with mean      
        """
        df = pd.read_csv(self.csv_file_path)
        df[column_name].fillna(df[column_name].mean(), inplace=True)
    
    def get_min_and_max_value_from_data(self, target_column:str)->tuple[float, float]:
        """
        This function return the min and max value from the selected column\n
        Argument:\n
            target_column : 
        Return:\n
            min_value : It returns the min value from the selected column in float
            max_value : It returns the max value from the selected column in float 
        """
        df = pd.read_csv(self.csv_file_path)
        min_value = df[target_column].min()
        max_value = df[target_column].max()
        return min_value, max_value
    
    def get_mean_absolute_division_of_column_from_dataset(self, column_name_1:str)->float:
        """
        This function get the mean absolute division value of column for further comparison\n
        Which ever mean value comes less select that dataset\n
        Argument:\n
            column_name_1: It takes the string value, choose the column from dataset
        Return:\n
            mean_absolute_division_first_column : It returns the float value of first column
        """
        df = pd.read_csv(self.csv_file_path)
        mean_first = df[column_name_1].mean(column_name_1)
        mean_absolute_division_first_column = df[column_name_1].sum(abs(column_name_1-mean_first))/len(column_name_1)
        return mean_absolute_division_first_column
    
    def get_standard_deviation_and_variance_of_columns_from_dataset(self, column_name_1:str)->tuple[float, float]:
        """
        This function calculate standard deviation of column from dataset\n
        This helpful when want to select only one dataset out of two, so that dataset can be more precise and less time consume when to predict\n
        Take the lowest value from the SD and variance values\n
        Argument:\n
            column_name_1: It takes the string value, choose the column from dataset
        Returns:\n
            first_column_sd : It returns the float value of first column SD
            first_column_variance : It returns the float value of first column variance
        """
        df = pd.read_csv(self.csv_file_path)
        first_column_sd = df[column_name_1].std(column_name_1)
        first_column_variance = df[column_name_1].var(column_name_1)
        return first_column_sd, first_column_variance

    def get_percentile_of_group_in_column_from_dataset(self, column_name:str, percentile_value:float)->float:
        """
        This function gives the percentile, means which if 50% provided then it gives its which group lies between this values\n
        Suppose age group is lies between 20 to 80, then it gives values which group ages people are more among\n
        Arguments:\n
            column_name: It takes the string value, choose the column from dataset
            percentile_value : It takes the float value (from 0 to 100)
        Return:\n
            percentile_range_value : It gives the which values lie in this percentile given range
        """
        df = pd.read_csv(self.csv_file_path)
        percentile_range_value = np.percentile(df[column_name], percentile_value)
        return percentile_range_value




