import Machine_Learning_algorithms_data.mahine_learning_algorithm as ml_algorithm
import Machine_Learning_algorithms_data.data_information as data_info

class BasciConcpetsFileConnection():

    def __init__(self, csv_path, X_dataset, y_dataset):
        self.csv_path = csv_path
        self.X_dataset = X_dataset
        self.y_dataset = y_dataset

        self.data_information = data_info.DataDescription(self.csv_path)
        self.machine_learning_algorithm = ml_algorithm.LinearRegrssionAlgorithms(self.X_dataset, self.y_dataset)
