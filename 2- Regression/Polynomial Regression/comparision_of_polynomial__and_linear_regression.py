
# Polynomial Regression

## Importing the libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## Importing the dataset"""

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:, -1].values

"""## Training the Linear Regression model on the whole dataset"""

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

"""## Training the Polynomial Regression model on the whole dataset"""

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg_2= LinearRegression()
lin_reg_2.fit(x_poly,y)

"""## Visualising the Linear Regression results"""

plt.scatter(x,y, color = 'red') 
plt.plot(x,lin_reg.predict(x), color ='blue')
plt.title('Salary/Level')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

"""## Visualising the Polynomial Regression results"""

plt.scatter(x,y, color = 'red') 
plt.plot(x,lin_reg_2.predict(x_poly), color ='blue')
plt.title('Salary/Level Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

"""## Predicting a new result with Linear Regression"""

lin_reg.predict([[6.5]])

"""## Predicting a new result with Polynomial Regression"""

lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))

