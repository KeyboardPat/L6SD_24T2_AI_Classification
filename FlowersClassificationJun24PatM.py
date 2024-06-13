import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load

#Import the dataset
data = pd.read_csv('iris.csv')

#Input and output
x = data.drop(['variety'], axis = 1)
y= data['variety']

#Scale input
stsc = StandardScaler()

x_scaled = stsc.fit_transform(x)

#String does NOT fit

#Train test splitting
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.2, random_state = 42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

#Initialise models
knc = KNeighborsClassifier()
dtc = DecisionTreeClassifier()
svc = SVC()

#Train models
knc.fit(x_train, y_train)
dtc.fit(x_train, y_train)
svc.fit(x_train, y_train)

#Test models
knc_preds = knc.predict(x_test)
dtc_preds = dtc.predict(x_test)
svc_preds = svc.predict(x_test)

#Evaluate model performance
knc_acc = accuracy_score(y_test, knc_preds)
dtc_acc = accuracy_score(y_test, dtc_preds)
svc_acc = accuracy_score(y_test, svc_preds)

#Evaluate model accuracy
models = [knc, dtc, svc]
model_acc_sc = [knc_acc, dtc_acc, svc_acc]
acc_model_index = model_acc_sc.index(max(model_acc_sc))
acc_model_object = models[acc_model_index]

#Save the most accurate model
dump(acc_model_object, "acc_model.joblib")

#Load model
loaded_model = load("acc_model.joblib")

#Manually enter data, test saved model 
sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))

#Make prediction
y_test1 = stsc.transform([[sepal_length, sepal_width, petal_length, petal_width]])

pred_value = loaded_model.predict(y_test1)
print(pred_value)








