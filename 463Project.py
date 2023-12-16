#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('/Users/senaoz/Documents/PycharmProjects/HouseSales/kc_house_data.csv')
data['date'] = pd.to_datetime(data['date'].str.slice(0, 8), format='%Y%m%d')
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data.drop(['id', 'date'], axis=1, inplace=True)

X = data.drop('price', axis=1)
y = data['price']


# In[2]:


def train_model():
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def predict(input_data):
    input_data = np.array(input_data)
    input_data = input_data.reshape(1, -1)
    input_data = poly.transform(input_data)
    
    if len(input_data[0]) != len(X_train[0]):
        return 'Wrong input data'
    
    if model.predict(input_data)[0] < 0:
        return 0
     
    else: 
        return model.predict(input_data)[0]
    


# In[3]:


input_data = [
  3, 
  1, 
  1180, 
  5650, 
  1,
  0, 
  0,
  3, 
  7, 
  1180, 
  0, 
  1955, 
  0, 
  98178, 
  47.5112, 
  -122.257, 
  1340, 
  5650, 
  2014,
  10, 
  13, 
];

predict(input_data)


# In[4]:


y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[5]:


mse, r2


# In[6]:


# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# In[7]:


# evaluate predictions
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MSE: %.2f%%" % (mse))


# In[8]:


accuracy = model.score(X_test, y_test)


# In[9]:


accuracy


# In[10]:


df = pd.read_csv('./kc_house_data.csv')
df.dataframeName = 'kc_house_data'
df


# In[11]:


data


# In[12]:


data.info()


# In[13]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dfCorr = df.copy()
dfCorr = dfCorr.drop(['id', 'date', 'zipcode'], axis=1)
numerical_columns = dfCorr.select_dtypes(include=['float64', 'int64'])
corr_matrix = numerical_columns.corr()

f, ax = plt.subplots(figsize=(12, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr_matrix, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()


# In[14]:


# make outlier analysis for df, show graphs

dfOutlier = df.copy()
dfOutlier = dfOutlier.drop(['id', 'date', 'zipcode'], axis=1)

numerical_columns = dfOutlier.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_columns.drop(['price'], axis=1)

for i in numerical_columns.columns:
    sns.boxplot(x=dfOutlier[i])
    plt.show()
    



# In[15]:


# make feature engineering, show graphs in a table format

dfFeature = df.copy()
dfFeature = dfFeature.drop(['id', 'date', 'zipcode'], axis=1)

numerical_columns = dfFeature.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_columns.drop(['price'], axis=1)

for i in numerical_columns.columns:
    sns.distplot(dfFeature[i])
    plt.show()
    


# In[16]:


# make analysis for the model, show graphs in a table format
y_pred_train = model.predict(X_train)
print("Quality Test {}".format(mean_squared_error(y_train, y_pred_train)))
y_pred = model.predict(X_test)
print("Quality Control {}".format(mean_squared_error(y_test, y_pred)))


# In[17]:


# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MSE: %.2f%%" % (mse))
print("R2: %.2f%%" % (r2))


# In[18]:


# Get absolute values of coefficients
abs_coefficients = abs(model.coef_)

# Create a dictionary mapping feature names to their absolute coefficients
feature_importance = dict(zip(data.columns, abs_coefficients))

# Sort the features by their importance
sorted_feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

# Print or visualize the feature importance
for feature, importance in sorted_feature_importance.items():
    print(f"{feature}: {importance}")


# In[18]:


# Plot the feature importance
plt.figure(figsize=(20, 10))
plt.bar(sorted_feature_importance.keys(), sorted_feature_importance.values())
plt.xticks(rotation=90)
plt.show()


# In[38]:


# Select a single data point from the test set for prediction
data_point = X_test[0]
prediction = model.predict(data_point.reshape(1, -1))

print("Prediction: {}".format(prediction[0]))
print("Actual Value: {}".format(y_test.iloc[0]))
print("Error: {}".format(prediction[0] - y_test.iloc[0]))
print("Mean Squared Error: {}".format(mean_squared_error(y_test, model.predict(X_test))))
print("R2 Score: {}".format(r2_score(y_test, model.predict(X_test))))
print("Accuracy: {}".format(model.score(X_test, y_test)))


# In[38]:




