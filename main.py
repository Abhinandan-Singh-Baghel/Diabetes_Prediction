import numpy as numpy
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


#data collection and analysis

diabetes_dataset = pd.read_csv('diabetes.csv')

# print(diabetes_dataset.head())

# diabetes_dataset.shape

# print(diabetes_dataset.describe())

diabetes_dataset['Outcome'].value_counts()

print(diabetes_dataset.groupby('Outcome').mean())


# seperating the data and Labels

X = diabetes_dataset.drop(columns = 'Outcome' , axis = 1)
Y = diabetes_dataset['Outcome']

print(X)
print(Y)


# data standardization

scalar = StandardScaler()

scalar.fit(X)

standardized_data = scalar.transform(X)

# print(standardized_data)

X = standardized_data
Y = diabetes_dataset['Outcome']


# Train Test Split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)


print(X.shape, X_train.shape, X_test.shape)


# Training the model

classifier = svm.SVC(kernel='linear')


#training the support vector machine classifier

classifier.fit(X_train, Y_train)



#model evaluation

# Accuracy score

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


print('Accuracy on training data : ', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on test data : ', test_data_accuracy)



# making a predective system

input_data = (6,148,72,35,0,33.6,0.627,50)

# input_data to numpy array

input_data_as_numpy_array = numpy.asarray(input_data)

# reshape the array as we are predicting for one instance
# two dimensional array

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

print(input_data_reshaped)

#standardize the input data 

std_data = scalar.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)


if(prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')







