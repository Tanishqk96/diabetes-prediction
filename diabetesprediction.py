import pandas as pd
import numpy as np

diabetes_data = pd.read_csv('/content/diabetes.csv')

diabetes_data.head()

diabetes_data.describe()

diabetes_data['Outcome'].value_counts()

diabetes_data.groupby('Outcome').mean()

x=diabetes_data.drop(columns = 'Outcome', axis =1)
y=diabetes_data['Outcome']

print(x)
print(y)

from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data and transform it
standardized_data = scaler.fit_transform(x)

x = standardized_data
y = diabetes_data['Outcome']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

from sklearn.svm import SVC

class logistic_regression():


  # declaring learning rate & number of iterations (Hyperparametes)
  def __init__(self, learning_rate, no_of_iterations):

    self.learning_rate = learning_rate
    self.no_of_iterations = no_of_iterations



  # fit function to train the model with dataset
  def fit(self, X, Y):

    # number of data points in the dataset (number of rows)  -->  m
    # number of input features in the dataset (number of columns)  --> n
    self.m, self.n = X.shape


    #initiating weight & bias value

    self.w = np.zeros(self.n)

    self.b = 0

    self.X = X

    self.Y = Y


    # implementing Gradient Descent for Optimization

    for i in range(self.no_of_iterations):
      self.update_weights()



  def update_weights(self):

    # Y_hat formula (sigmoid function)

    Y_hat = 1 / (1 + np.exp( - (self.X.dot(self.w) + self.b ) ))


    # derivaties

    dw = (1/self.m)*np.dot(self.X.T, (Y_hat - self.Y))

    db = (1/self.m)*np.sum(Y_hat - self.Y)


    # updating the weights & bias using gradient descent

    self.w = self.w - self.learning_rate * dw

    self.b = self.b - self.learning_rate * db


  # Sigmoid Equation & Decision Boundary

  def predict(self, X):

    Y_pred = 1 / (1 + np.exp( - (X.dot(self.w) + self.b ) ))
    Y_pred = np.where( Y_pred > 0.5, 1, 0)
    return Y_pred

#svm_classifier = SVC(kernel='linear' )
#svm_classifier.fit(x_train, y_train)

classifier = logistic_regression(learning_rate = 0.02, no_of_iterations=1111)

classifier.fit(x_train, y_train)

from sklearn.metrics import accuracy_score

# THIS IS THE ACCURACY DATA FOR LOGISTIC MODEL
train_predictions = classifier.predict(x_train)
train_accuracy = accuracy_score(train_predictions,y_train )
print('the accuracy of the training data is :',train_accuracy )

# THIS IS THE ACCURACY DATA FOR LOGISTIC MODEL
test_predictions = classifier.predict(x_test)
test = accuracy_score(test_predictions,y_test )
print('the accuracy of the training data is :',test )

# THIS IS THE ACCURACY DATA FOR SVM CLASSIFIER MODEL
train_predictions = svm_classifier.predict(x_train)
train_accuracy = accuracy_score(train_predictions,y_train )
print('the accuracy of the training data is :',train_accuracy )

# THIS IS THE ACCURACY DATA FOR SVM CLASSIFIER MODEL
test_predictions = svm_classifier.predict(x_test)
test = accuracy_score(test_predictions,y_test )
print('the accuracy of the training data is :',test )

input_data = (6	,148	,72	,35	,0	,33.6,	0.627,	50)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('THE PERSON IS NOT DIABETIC')
else:
  print('THE PERSON IS DIABETIC')

import matplotlib.pyplot as plt
diabetes_data.hist(figsize=(10, 10), bins=20)
plt.suptitle('Feature Distribution')
plt.show()

