# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries. 

2.Set variables for assigning dataset values. 

3.Import linear regression from sklearn. 

4.Assign the points for representing in the graph. 

5.Predict the regression for marks by using the representation of the graph. 

6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 212222040135
RegisterNumber: Roghith K 
*/
import pandas as pd
df=pd.read_csv('/content/$ - Sheet1 (2).csv')
df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/$ - Sheet1 (2).csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.xlabel('Y')
X=df.iloc[:,0:1]
Y=df.iloc[:,-1]
Y
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(X_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.xlabel('Y')
plt.plot(X_train,lr.predict(X_train),color='red')


```

## Output:
![image](https://github.com/selva258963/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121961701/4e6125b3-4af6-4725-b6f5-55f001f799a9)
![image](https://github.com/selva258963/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121961701/af8d6442-152a-46dd-a26d-37da9291b809)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
