# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Gather data consisting of two variables. Input- a factor that affects the marks and Output - the marks scored by students

2.Plot the data points on a graph where x-axis represents the input variable and y-axis represents the marks scored

3.Define and initialize the parameters for regression model: slope controls the steepness and intercept represents where the line crsses the y-axis

4.Use the linear equation to predict marks based on the input Predicted Marks = m.(hours studied) + b

5.for each data point calculate the difference between the actual and predicted marks

6.Adjust the values of m and b to reduce the overall error. The gradient descent algorithm helps update these parameters based on the calculated error

7.Once the model parameters are optimized, use the final equation to predict marks for any new input data 

## Program:
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: REVATHI K

RegisterNumber: 212223040169

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import libraries to find mae, mse
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

#read csv file
df= pd.read_csv('data.csv')

#displaying the content in datafile
df.head()
df.tail()

# Segregating data to variables
X=df.iloc[:,:-1].values
X
y=df.iloc[:,-1].values
y

#splitting train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/2,random_state=0)

#import linear regression model and fit the model with the data
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#displaying predicted values
y_pred=regressor.predict(X_test)
y_pred

#displaying actual values
y_test

#graph plot for training data
import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

#graph plot for test data
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,regressor.predict(X_test),color='blue')
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")

#find mae,mse,rmse
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)

```

## Output:
Head Values

![Screenshot 2024-08-16 154352-1](https://github.com/user-attachments/assets/7797950e-b9fe-4c05-a540-dd2195843bef)

Tail Values

![Screenshot 2024-08-16 154419-1](https://github.com/user-attachments/assets/939b8757-9cdc-44a6-9342-b81b52197ff0)

X Values

![Screenshot 2024-08-16 152702](https://github.com/user-attachments/assets/837ea215-e4d0-4455-bf09-287f50e42930)

Y Values

![Screenshot 2024-08-16 153116-1](https://github.com/user-attachments/assets/0eb9b8d0-8f5f-4941-a0ac-4bd10b974a48)

Predicted Values

![Screenshot 2024-08-16 161908](https://github.com/user-attachments/assets/29b2e906-766f-49b9-8721-ebee64b58f54)

Actual Values

![Screenshot 2024-08-16 153301](https://github.com/user-attachments/assets/8ace62ad-d5ed-45b1-b9f4-f80cded316a1)

Testing Set

![download (8)](https://github.com/user-attachments/assets/720dbcc4-65a7-487b-b670-564cfc008e87)

Training Set

![download (7)-1](https://github.com/user-attachments/assets/9e34bc8f-d554-4b9a-a8cf-f13d48bc9710)

MSE,MAE and RMSE

![Screenshot 2024-08-16 153958-1](https://github.com/user-attachments/assets/6ab5c148-6d20-49a8-b0de-2b345089f924)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
