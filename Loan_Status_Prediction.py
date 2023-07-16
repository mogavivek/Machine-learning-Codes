import numpy as np
import Machine_Learning_algorithms_data.data_information as data_info
import Machine_Learning_algorithms_data.mahine_learning_algorithm as ml_algorithm


csv_path = 'C:/Users/vivek/PycharmProjects/pythonProject/Vivekcode/Machine-learning-Codes/data-set/train_u6lujuX_CVtuZ9i (1).csv'

loan_dataset_df = data_info.DataDescription(csv_path)
loan_dataset, loan_dataset_head, loan_dataset_describe, loan_dataset_info, loan_dataset_null_values = loan_dataset_df.data_description()

print(loan_dataset_head)
print(loan_dataset_null_values)

#Now dropping the missing values
loan_dataset = loan_dataset.dropna()
print(loan_dataset.isnull().sum())

#Now replacing the loan status in numerical value by using label encoding
loan_dataset.replace({"Loan_Status":{"N":0, "Y":1}}, inplace=True)
print(loan_dataset.head())
print(loan_dataset["Loan_Status"])

#Counting the dependent total values
print(loan_dataset['Dependents'].value_counts())

#Now replacing the value of 3+ in 4 in Dependents column
loan_dataset = loan_dataset.replace(to_replace="3+", value=4)
print(loan_dataset['Dependents'].value_counts())

"""Data Visulization"""
#educaiton and loan status
data_info.DataDescription.plot_classification_in_seaborn(loan_dataset, 'Education', 'Loan_Status')

#visulization for merital and loan status
data_info.DataDescription.plot_classification_in_seaborn(loan_dataset, 'Married', 'Loan_Status')

#Convert categorical columns to numerical values
loan_dataset.replace({"Married":{"No":0, "Yes":1}, 
                      "Gender":{"Male":1, "Female":0}, 
                      "Self_Employed":{"No":0, "Yes":1}, 
                      "Property_Area":{"Rural":0, "Semiurban":1, "Urban":2},
                      "Education":{"Graduate":1, "Not Graduate":0}}, inplace=True)

print(loan_dataset.head())

#Now separating the dataset in x and y
X = loan_dataset.drop(columns=["Loan_ID", "Loan_Status"], axis=1)
y = loan_dataset["Loan_Status"]

svm_model_dataset = ml_algorithm.LinearRegrssionAlgorithms(X, y)
model, training_data_accuracy, test_data_accuracy = svm_model_dataset.support_vector_machine_algorithem(
                                                "classification problem", 0.1, 2)

print("Accuracy on the training dataset: ",training_data_accuracy)
print("Accuracy on the test dataset: ",test_data_accuracy)

# making a predicitve system



