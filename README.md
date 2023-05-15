# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

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
// file:///home/sec/Pictures/Screenshot%20from%202023-05-15%2014-02-34.png![image](https://github.com/Yugendaran/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128135616/aad92dda-d4ee-40ee-8e3a-a747e72b22bf)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
