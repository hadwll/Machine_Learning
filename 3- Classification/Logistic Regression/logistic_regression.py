
# Logistic Regression

## Importing the libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## Importing the dataset"""

dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

"""## Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

"""## Feature Scaling"""

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.transform(x_test)

"""## Training the Logistic Regression model on the Training set"""

from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state =0)
classifier.fit(x_train,y_train)

"""## Predicting a new result"""

print(classifier.predict(sc.transform([[30,87000]])))

"""## Predicting the Test set results"""

y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_pred),1)),1))

"""## Making the Confusion Matrix"""

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print(cm)
print(acc)