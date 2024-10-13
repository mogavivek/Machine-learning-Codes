import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class SNSChart():

    def __init__(self: "SNSChart", csv_file_path:str)->None:
        """
        This class store the different SNS chart procedure to get more clear idea about the dataset\n
        In red it will shows the mean value of selected column and in green shows the median value of selected column
        """
        self.csv_file_path = csv_file_path
    
    def hist_plot(self, target_column:str, min_value:int, max_value:int, offset:int, show_median_value:bool=True, show_mean_value:bool = True)->None:
        """
        This function generate the chart between the range and show the mean value\n
        Arguments:\n
            target_column : It takes the string value of column name
            min_value : It takes the integer value
            max_value : It takes the integer value
            offset : It takes the integer value
            show_median_value (Optional) : It takes the boolean value, if want this data then True (ByDefault) else False
            show_mean_value (Optional) : It takes the boolean value, if want this data then True (ByDefault) else False
        """
        dataset = pd.read_csv(self.csv_file_path)
        sns.histplot(x=target_column, data=dataset, bins=[i for i in range(min_value, max_value, offset)])
        if show_mean_value is True:
            mn = np.mean(dataset[target_column]) 
            plt.plot([mn for i in range(0, 300)], [i for i in range(0, 300)], c="red", label="mean")
            plt.show()
        if show_median_value is True:
            mn = np.median(dataset[target_column]) 
            plt.plot([mn for i in range(0, 300)], [i for i in range(0, 300)], c="green", label="median")
            plt.show()
    
    def check_mode_of_categorical_column_in_chart(self, target_column_for_data_chart:str, target_column_for_mode:str, min_value:int, max_value:int, offset:int):
        """
        This function use to check the categorical data of selected column
        Arguments:\n
            target_column_for_data_chart : It takes the string value of column name which chart want to see
            target_column_for_mode : It takes the string value of column name which want to see mode in a chart
            min_value : It takes the integer value
            max_value : It takes the integer value
            offset : It takes the integer value
        """
        dataset = pd.read_csv(self.csv_file_path)
        mode = dataset[target_column_for_mode].mode()[0]
        sns.histplot(x=target_column_for_data_chart, data=dataset, bins=[i for i in range(min_value, max_value, offset)])
        plt.plot([mode for i in range(0, 300)], [i for i in range(0, 300)], c="green", label="mode")
        plt.show()
    
    def get_scatter_chart_with_compare_to_mean_value(self, column_name_1:str, column_name_2:str, first_name:str, second_name:str)->None:
        """
        This function check the scatter chart of two dataset columns\n
        Then after check that the which values are near to the mean absolute division that column can be selected for further use\n
        Arguments:\n
            column_name_1: It takes the string value, choose the column from dataset
            column_name_2: It takes the string value, choose the second column from the dataset
            first_name : It takes the string value and provide name which will visible in chart for first selected column points
            second_name : It takes the string value and provide name which will visible in chart for second selected column points 
        """
        df = pd.read_csv(self.csv_file_path)
        row, col = df[column_name_1].shape
        mean = df[column_name_1].mean(column_name_1)
        plt.figure(figsize=(10, 3))
        plt.scatter(column_name_1, label=first_name)
        plt.scatter(column_name_2, label=second_name)
        plt.plot([mean for i in range(1, row)], row, c="blue", label="mean")
        plt.show()
    
    def check_correlation_or_covariance_graph(self, dataset_correlation:tuple)->None:
        """This function check the correlation graph of selected dataset\n
        Argument:
            dataset_correlation : It takes the input of correlation or covariance dataset
        """
        plt.figure(figsize=(4, 3))
        sns.heatmap(dataset_correlation, annot=True)
        plt.show()



    


    
