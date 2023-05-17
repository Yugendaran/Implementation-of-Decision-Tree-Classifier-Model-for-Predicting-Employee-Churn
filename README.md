# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import standard libraries in python for finding Decision tree classsifier model for predicting 
employee churn.

2.Initialize and print the Data.head(),data.info(),data.isnull().sum()
3.visualize data value count.
4.import sklearn from LabelEncoder.
5.split data into training and testing.  
6.Calculate the accuracy, data prediction by importing the required modules from sklearn


## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: YUGENDARAN.G
RegisterNumber:  212221220063

import pandas as pd
data=pd.read_csv("/content/Employee.csv")

print("data.head():")
data.head()

print("data.info():")
data.info()

print("isnull() and sum():")
data.isnull().sum()

print("data value counts():")
data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()

print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])

```

## Output:
![image](https://github.com/Yugendaran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135616/b1162149-bbea-43a7-96a9-354fd108a151)

![image](https://github.com/Yugendaran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135616/bfe60847-ed9a-487c-a700-b43dca8659a5)

![image](https://github.com/Yugendaran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135616/7f03738f-ea1f-4338-a98b-1ef247f52708)

![image](https://github.com/Yugendaran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135616/a4e857d7-6fa3-4dc1-b615-ec4f8433c511)

![image](https://github.com/Yugendaran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135616/ff85899e-0a75-4138-9b2a-d9abcdf20138)

![image](https://github.com/Yugendaran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135616/6bf35c61-a89f-4af3-bdaf-840d8b320af5)

![image](https://github.com/Yugendaran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135616/3010e840-d901-478a-ba93-82eba77e27e9)

![image](https://github.com/Yugendaran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135616/ebb42f60-6046-426c-9759-e629b2a68b2c)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
