'''                     Import libraries.                       '''
import polars as pl
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

import os
import sys
# import requests
import numpy as np
import pandas as pd

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


'''                     Load and analyse the data.                       '''
# Load the dataset.
data_iris = load_iris()
print(data_iris['DESCR'])

# Create a Polars DataFrame from the iris dataset.
'''
X, y = load_iris(return_X_y=True, as_frame=True)

Note: In Polars, we use the `DataFrame` constructor to create a new dataframe.
The `schema` parameter is used to specify the column names.
The data is in the `data` attribute and the feature names are in `feature_names`.
The `target` attribute contains the target variable.

Note: In Polars, we use the `with_columns` method to add a new column.
'''
df_iris = pl.DataFrame(data_iris.data, schema=data_iris.feature_names)
df_iris = df_iris.with_columns(pl.Series('target', data_iris.target))
print(df_iris.head())

# Target mapping.
target_mapping = {
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
}

### Scatter plot the data.
## Sepal length vs Sepal width
# Plot each class separately
for target_class in df_iris['target'].unique():
    plt.scatter(
        x=df_iris.filter(pl.col('target') == target_class)['sepal length (cm)'],
        y=df_iris.filter(pl.col('target') == target_class)['sepal width (cm)'],
        s=100,
        label=target_mapping[target_class],  # Add a label for each class
        cmap='viridis',
        edgecolor='k',
        alpha=0.7,
        marker='X'
    )

# Add labels and title
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris Dataset Scatter Plot')
plt.legend()  # Display the legend
plt.show()

## Petal length vs Petal width
# Plot each class separately
for target_class in df_iris['target'].unique():
    plt.scatter(
        x=df_iris.filter(pl.col('target') == target_class)['petal length (cm)'],
        y=df_iris.filter(pl.col('target') == target_class)['petal width (cm)'],
        s=100,
        label=target_mapping[target_class],  # Add a label for each class
        cmap='viridis',
        edgecolor='k',
        alpha=0.7,
        marker='X'
    )

# Add labels and title
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Iris Dataset Scatter Plot')
plt.legend()  # Display the legend
plt.show()



