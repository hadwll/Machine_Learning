

# Decision Tree Regression

## Importing the libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## Importing the dataset"""

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

"""## Training the Decision Tree Regression model on the whole dataset"""

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x,y)

"""## Predicting a new result"""

regressor.predict([[6.5]])

"""## Visualising the Decision Tree Regression results (higher resolution) only  possible for 1  feature."""

x_grid = np.arange(min(x), max(x),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x,y, color = 'red') 
plt.plot(x_grid,regressor.predict(x_grid),color ='blue')
plt.title('Salary/Level')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()