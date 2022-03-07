import streamlit as st

import numpy as np
import pandas as pd
import numpy as np
import sklearn as st
!pip install sweetviz
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.header("Housing Price Prediction")
df = pd.read_csv('train.csv')
df.head()
df.info()

# importing sweetviz
import sweetviz as sv


#analyzing the dataset
advert_report = sv.analyze(df)


#display the report
advert_report.show_html('Bulding type price.html')
advert_report.show_notebook()

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_df.head()
test_df.head()
comparison_report = sv.compare([train_df,'Train'], [test_df,'Test'], target_feat='SalePrice')
comparison_report.show_notebook()
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model

# Taking care of missing data

# First find out all the columns that have missing data and arrange in descending order
total = train_df.isna().sum().sort_values(ascending=False)

# concatenate this data into dataframe
missing_data = pd.concat([total], axis=1, keys=["Total"])

# dropping the columns where missing data is more than 1
train_df = train_df.drop((missing_data[missing_data.get("Total") > 1]).index, 1)

# Drop the row entry
train_df = train_df.drop(train_df.loc[train_df.get("Electrical").isna()].index)
train_df.isna().sum().max()

# Encoding the categorical variables with one hot encoding

# First getting all the columns with categories
categories = list(train_df.select_dtypes(["object"]))

# Applying one hot encoding 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categories)], remainder='passthrough')

X = train_df.drop(['Id', 'SalePrice'], axis=1)

X

print(X.shape)
test_df = pd.read_csv('test.csv')
test_df = test_df.drop((missing_data[missing_data.get("Total") > 1]).index, 1)
print(test_df.shape)



#
