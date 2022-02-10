

# Support Vector Regression (SVR)

## Importing the libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## Importing the dataset"""

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

"""# reshape the y  vector"""

y = y.reshape([len(y),1])

"""## Feature Scaling"""

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()

x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)

"""## Training the SVR model on the whole dataset

"""

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x, y)

"""## Predicting a new result"""

sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1))

"""## Visualising the SVR results"""

plt.scatter(sc_x.inverse_transform(x),sc_y.inverse_transform(y), color = 'red') 
plt.plot(sc_x.inverse_transform(x),sc_y.inverse_transform(regressor.predict(x).reshape(-1,1)), color ='blue')
plt.title('Salary/Level SVR Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

