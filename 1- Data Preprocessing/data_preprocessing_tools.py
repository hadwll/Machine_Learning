

# Data Preprocessing Tools

## Importing the libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## Importing the dataset"""

dataset =pd.read_csv('Data.csv')
##matrix of features select all but the last column
x = dataset.iloc[:,:-1].values
##dependant variable vector  selects the last column
y = dataset.iloc[:, -1].values

"""## Taking care of missing data"""

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
## fit the data in columns 1-3
imputer.fit(x[:,1:3])
# replace the missing data with the mean values
x[:,1:3] = imputer.transform(x[:,1:3])

"""## Encoding categorical data

##Encoding the Independent Variable
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

"""### Encoding the Dependent Variable"""

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

"""## Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(x,y, test_size =0.2, random_state = 1 )

"""## Feature Scaling"""

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train[:,3:]= sc.fit_transform(x_train[:,3:])
x_test[:,3:]= sc.transform(x_test[:,3:])

print(x_train)

print(x_test)

