# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.

2. Set variables for assigning dataset values.

3. Import linear regression from sklearn.

4. Assign the points for representing in the graph.

5. Predict the regression for marks by using the representation of the graph.

6. Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/student_scores.csv')
#displaying the content in datafile
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
y_pred
y_test

y_pred

#graph plot for training data
plt.scatter(x_train,y_train,color="darkseagreen")
plt.plot(x_train,regressor.predict(x_train),color="plum")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(x_test,y_test,color="darkblue")
plt.plot(x_test,regressor.predict(x_test),color="plum")
plt.title("Hours vs Scores (Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
![ml 1. df.head()](https://github.com/RoghithKrishnamoorthy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475474/ca235bb0-e868-4a38-9893-15601a1bc95d)
![ml 2.df.tail()](https://github.com/RoghithKrishnamoorthy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475474/b5909b34-832d-459e-b194-19802194e875)
![ml 3. Array value of X](https://github.com/RoghithKrishnamoorthy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475474/ee7ac194-8bbe-4477-8942-a4692ec4affa)
![ml 4. Array value of Y](https://github.com/RoghithKrishnamoorthy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475474/006abdc4-2aa4-4455-8ac7-21a539832c11)
![ml 5. Values of Y prediction](https://github.com/RoghithKrishnamoorthy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475474/f3c5ad6f-06c9-4e3b-a59b-c0ccae7ee758)
![ml 6. Array values of Y test](https://github.com/RoghithKrishnamoorthy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475474/16ed07c2-d468-4388-85bd-8a992f26cd46)
![ml 7. Training Set Graph](https://github.com/RoghithKrishnamoorthy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475474/eec638a1-76ac-4451-9424-b7f386046686)
![ml 8. Training Set Graph](https://github.com/RoghithKrishnamoorthy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475474/afc6e051-9ea2-4191-be8e-259bc8d69a71)
![ml 9. Values of MSE, MAE and RMSE](https://github.com/RoghithKrishnamoorthy/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475474/42cd0021-dda0-4894-9a7b-7688d6438166)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
