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
    
    def fill_na_with_mean_values_of_selected_column(self, column_name:str):
        """This function fill the null values with the mean values\n
        Arguments:\n
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
    
    def check_skewness(self, column_name:str)->float:
        """
        This function check the skewness the target column
        If the value comes in negative then it is negative skewness and positive then positive skewness\n
        If there are no skewness means normal skewness then it returns the zero value
        Arguments:\n
            column_name: It takes the string value, choose the column from dataset
        Return:\n
            skewness: It returns the float values
        """
        df = pd.read_csv(self.csv_file_path)
        skewness = df[column_name].skew()
        return skewness

    def check_correlation_of_dataset(self, float_datatypes:str, int_datatypes:str)->tuple:
        """
        This function check the correlation where if the correlation is 1 to 0 then it is positive correlation\n
        If the value is 0 then neutral correlation and if the value lies between 0 to -1 then negative correlation\n 
        positive correlation if x value increase then y will also increase, vice-versa for negative correlation\n
        If the value x is increasing or decreasing, but the y is not changing then it is neutral\n
        Arguments:\n
            float_datatypes : first check in the dataset which types of float values are float64, float32
            int_datatypes : first check in the dataset which types of int values are int64, int32
        """
        df = pd.read_csv(self.csv_file_path)
        data_corr = df.select_dtypes([float_datatypes, int_datatypes]).corr()
        return data_corr
    
    def check_covariance_of_dataset(self, float_datatypes:str, int_datatypes:str)->tuple:
        """
        This function check the covariance where
        Arguments:\n
            float_datatypes : first check in the dataset which types of float values are float64, float32
            int_datatypes : first check in the dataset which types of int values are int64, int32
        """
        df = pd.read_csv(self.csv_file_path)
        data_cova = df.select_dtypes([float_datatypes, int_datatypes]).cov()
        return data_cova

    def apply_method_central_limit_theorem(self, column_name:str, number_of_sample_want:int, ten_percent_of_data_value:int)->list[float]:
        """
        This function convert the non-normal-distributed data to normal-distribution data with the help of\n
        central limit theorem and plot in the graph\n
        Arguments:\n
            column_name: It takes the string value, choose the column from dataset
            number_of_sample_want : It takes the integer value, means if you want to take 50 samples from your data then give 50
            ten_percent_of_data_value : It takes the integer value, it should be more than 30 and 10% of your len(column_name)
        """
        df = pd.read_csv(self.csv_file_path)
        data_mean = np.mean(df[column_name])
        sample_mean = []
        for i in range(number_of_sample_want):
            sample_data = []
            for data in range(ten_percent_of_data_value):
                sample_data.append(np.random.choice(df[column_name]))
            sample_mean.append(np.mean(sample_data))
        
        sample_M = pd.DataFrame({"Sample_Mean": sample_mean})
        sample_data_mean = np.mean(sample_mean)
        print("The population data and sample data mean should be similar or near to each other")
        print(f"The population data mean is: {data_mean} and the sample data mean is {sample_data_mean}")
        plt.figure(figsize=(4, 3))
        sns.kdeplot(x="Sample_Mean", data=sample_M)
        plt.show()
    




