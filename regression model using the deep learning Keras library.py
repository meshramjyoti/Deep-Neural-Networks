#!/usr/bin/env python
# coding: utf-8

# # Regression model using the deep learning Keras library

# #### Objective: I am going to build a regression model using the Keras library to model the data about concrete compressive strength.
# 
# -The predictors in the data of concrete strength include:
# 
# 1. Cement
# 2. Blast Furnace Slag
# 3. Fly Ash
# 4. Water
# 5. Superplasticizer
# 6. Coarse Aggregate
# 7. Fine Aggregate

# # Download and Clean Dataset
# #### Let's start by importing the pandas and the Numpy libraries.

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()


# In[6]:


concrete_data.shape #data points 


# In[5]:


concrete_data.describe()


# In[7]:


concrete_data.isnull().sum() # CLEAN THE DATA


# ### Split data into predictors and target

# In[8]:


concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column


# In[9]:


predictors.head()


# In[10]:


target.head()


#  ### normalize the data by substracting the mean and dividing by the standard deviation.

# In[11]:


predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()


# ### save the number of predictors to n_cols

# In[13]:


n_cols = predictors_norm.shape[1] # number of predictors


# # Import the Keras library

# In[14]:


import keras


# In[15]:


from keras.models import Sequential
from keras.layers import Dense


# # Build a Neural Network

# #### Model that has one hidden layer with 10 neurons and a ReLU activation function. It uses the adam optimizer and the mean squared error as the loss function.

# In[17]:


# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# ### import scikit-learn in order to randomly split the data into a training and test sets
# 

# In[18]:


from sklearn.model_selection import train_test_split


# ### Splitting the data into a training and test sets by holding 30% of the data for testing
# 
# 

# In[20]:


X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=42)


# # Train and Test the Network

# In[21]:


# build the model
model = regression_model()


# ### Next, we will train and test the model at the same time using the fit method. We will leave out 30% of the data for validation and we will train the model for 50 epochs.

# In[23]:


# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=50, verbose=1)


# # Evaluate the model on the test data

# In[24]:


loss_val = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
loss_val


# ### compute the mean squared error between the predicted concrete strength and the actual concrete strength.
# 
# Let's import the mean_squared_error function from Scikit-learn.

# In[25]:


from sklearn.metrics import mean_squared_error


# In[26]:


mean_square_error = mean_squared_error(y_test, y_pred)
mean = np.mean(mean_square_error)
standard_deviation = np.std(mean_square_error)
print(mean, standard_deviation)


# # Create a list of 50 mean squared errors and report mean and the standard deviation of the mean squared errors.

# In[27]:


total_mean_squared_errors = 50
epochs = 50
mean_squared_errors = []
for i in range(0, total_mean_squared_errors):
    X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.3, random_state=i)
    model.fit(X_train, y_train, epochs=epochs, verbose=0)
    MSE = model.evaluate(X_test, y_test, verbose=0)
    print("MSE "+str(i+1)+": "+str(MSE))
    y_pred = model.predict(X_test)
    mean_square_error = mean_squared_error(y_test, y_pred)
    mean_squared_errors.append(mean_square_error)

mean_squared_errors = np.array(mean_squared_errors)
mean = np.mean(mean_squared_errors)
standard_deviation = np.std(mean_squared_errors)


print("Below is the mean and standard deviation of " +str(total_mean_squared_errors) + " mean squared errors without normalized data. Total number of epochs for each training is: " +str(epochs) + "\n")
print("Mean: "+str(mean))
print("Standard Deviation: "+str(standard_deviation))


# # THANK YOU
