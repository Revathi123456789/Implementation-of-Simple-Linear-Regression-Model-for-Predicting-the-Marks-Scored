# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Store data in a structured format (e.g., CSV, DataFrame).
2. Use a Simple Linear Regression model to fit the training data.
3. Use the trained model to predict values for the test set.
4. Evaluate performance using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

## Program:
```
Developed by: Ramya P
RegisterNumber: 212223230168

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
#displaying the content in datafile
df.head()
![mlpic](https://github.com/user-attachments/assets/46372791-1b03-4fef-9bf9-21ac7f56215c)













## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
