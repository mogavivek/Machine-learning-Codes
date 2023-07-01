import numpy as np
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

#Now predicting the value
input_data = np.asarray(0.02,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.066,0.2273,0.31,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.555,0.6711,0.6415,0.7104,0.808,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.051,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.018,0.0084,0.009,0.0032)

#reshape the input data because it is predicting for one instance
input_data_reshape = input_data.reshape(1, -1)

prediction = model.predict(input_data_reshape)

if(prediction[0] == 'R'):
    print('The object is Rock')
else:
    print('The object is mine')
