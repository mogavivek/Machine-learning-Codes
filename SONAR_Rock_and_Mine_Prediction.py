import Machine_Learning_algorithms_data.mahine_learning_algorithm as ml_algorithm
import Machine_Learning_algorithms_data.data_information as data_info

dataset_path = "C:/Users/vivek/PycharmProjects/pythonProject/Vivekcode/Machine-learning-Codes/data-set/Copy of sonar data.csv"

#Data collection and data processing
sonar_data = data_info.DataDescription(dataset_path)
sonar_data_df, sonar_data_headvalue, sonar_data_description, sonar_data_info,  sonar_data_null_values = sonar_data.data_description()
print(sonar_data_headvalue)
print(sonar_data_description)
print(sonar_data_info)
print(sonar_data_null_values)

#60 shows the column of M and R
print(sonar_data_df.iloc[:,60].value_counts())

#Splitting in x and y
X = sonar_data_df.iloc[:, :60]
print(X.shape)
y = sonar_data_df.iloc[:,60]
print(y.shape)

the_logitic_model = ml_algorithm.LinearRegrssionAlgorithms(X, y)
model, training_score, test_score = the_logitic_model.logistic_regression(0.1, 1)

print("The accuracy score of the train data set: ", training_score)
print("The accuracy score of the test dataset: ", test_score)
