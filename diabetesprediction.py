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

svm_classifier = SVC(kernel='linear' )
svm_classifier.fit(x_train, y_train)

from sklearn.metrics import accuracy_score

train_predictions = svm_classifier.predict(x_train)
train_accuracy = accuracy_score(train_predictions,y_train )
print('the accuracy of the training data is :',train_accuracy )

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

prediction = svm_classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('THE PERSON IS NOT DIABETIC')
else:
  print('THE PERSON IS DIABETIC')

import matplotlib.pyplot as plt
diabetes_data.hist(figsize=(10, 10), bins=20)
plt.suptitle('Feature Distribution')
plt.show()

