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
        """This function generate the chart between the range and show the mean value"""
        dataset = pd.read_csv(self.csv_file_path)
        sns.histplot(x=target_column, data=dataset, bins=[i for i in range(min_value, max_value, offset)])
        if show_mean_value is True:
            mn = np.mean(dataset[target_column]) 
            plt.plot([mn for i in range(0, 300)], [i for i in range(0, 300)], c="red")
            plt.show()
        if show_median_value is True:
            mn = np.median(dataset[target_column]) 
            plt.plot([mn for i in range(0, 300)], [i for i in range(0, 300)], c="green")
            plt.show()
    
    def check_mode_of_categorical_column_in_chart(self, target_column_for_data_chart:str, target_column_for_mode:str, min_value:int, max_value:int, offset:int):
        """This function use to check the categorical data of selected column"""
        dataset = pd.read_csv(self.csv_file_path)
        mode = dataset[target_column_for_mode].mode()[0]
        sns.histplot(x=target_column_for_data_chart, data=dataset, bins=[i for i in range(min_value, max_value, offset)])
        plt.plot([mode for i in range(0, 300)], [i for i in range(0, 300)], c="green")
        plt.show()

    
