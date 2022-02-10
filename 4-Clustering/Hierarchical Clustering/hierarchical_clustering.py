
# Hierarchical Clustering

## Importing the libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## Importing the dataset"""

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

"""## Using the dendrogram to find the optimal number of clusters"""

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method ='ward'))

"""## Training the Hierarchical Clustering model on the dataset"""

from sklearn.cluster import AgglomerativeClustering
hc_cluster = AgglomerativeClustering(n_clusters=5)
y_pred=hc_cluster.fit_predict(X)

