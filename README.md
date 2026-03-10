# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required libraries and load the food dataset.
2.Select features and split the dataset into training and testing data with scaling.
3.Train the Support Vector Machine (SVM) model using GridSearchCV to find the best parameters.
4.Predict the results and evaluate the model using accuracy, classification report, and confusion matrix.
```
## Program:
```
/*
Program to implement SVM for food classification for diabetic patients.
Developed by: Arthi S
RegisterNumber:  212225220011
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('food_items_binary.csv')
print(data.head())
print(data.columns)

features = ['Calories','Total Fat','Saturated Fat','Sugars','Dietary Fiber','Protein']
target = 'class'
x = data[features]
y = data[target]

x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

svm = SVC()

param_grid = {
    'C': [0.1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale','auto']
}

grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train,y_train)

best_model = grid_search.best_estimator_
print("Name:Arthi S")
print("Register Number:212225220011")
print("Best Parameters:", grid_search.best_params_)


accuracy = accuracy_score(y_test, y_pred)
print("Name:Arthi S")
print("Register Number:212225220011")
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Output:

<img width="735" height="576" alt="Screenshot 2026-03-10 130904" src="https://github.com/user-attachments/assets/d4afaf59-48c5-4605-ae0a-5dd2bfdbb6a2" />



## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
