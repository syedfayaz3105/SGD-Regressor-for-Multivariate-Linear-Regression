# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.mport necessary libraries
2.Load and preprocess the data (define features and target).
3.Split the dataset into training and testing sets.
4.Scale the features using StandardScaler.
5.Train the SGDRegressor model on the training set.
6.Evaluate the model on both training and testing sets using MSE or other metrics.
```

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Farhana H
RegisterNumber: 212223230057 
*/
```
```
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the California Housing dataset
data=fetch_california_housing()

# Use the first 3 features as inputs
X = data.data[:,:3] # Features : 'MedInc' ,'House Age','AveRooms'

# Use 'MedHouseVal' and 'AveOccup' as output variables
Y = np.column_stack((data.target,data.data[:,6])) # Targets : 'Med HouseVal',AveOccup'

# Split the data into training and testing sets
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=42)

# scale the features and target variables
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
Y_train = scaler_Y.fit_transform(Y_train)
Y_test = scaler_Y.transform(Y_test)

# Initialize the SGDRegressor
sgd = SGDRegressor(max_iter=1000, tol=1e-3)

# Use MultiOutputRegressor to handle multiple output
multi_output_sgd = MultiOutputRegressor(sgd)

# Train the model
multi_output_sgd.fit(X_train, Y_train)

# Predict on the test data
Y_pred = multi_output_sgd.predict(X_test)

# Inverse transform the predictions to get them back to the orginal scale.
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)

# Evalulate the model using Mean Squared Error
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:",mse)

# Optionally, print some predictions
print("\nPredictions:\n",Y_pred[:5]) #Print tfirst 5 predictions
```

## Output:
![image](https://github.com/user-attachments/assets/d650fbf1-0a1b-45f8-825d-b8ff84a58edb)



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
