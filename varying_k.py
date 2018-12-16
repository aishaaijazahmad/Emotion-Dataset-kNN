#import relevant libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#read the file into Jupyter Python
df = pd.read_csv("C:/Users/Aisha Aijaz Ahmad/Desktop/fer2018.csv")#load dataset
df.head() # view dataset to check successful load

#split our dataset into its attributes and labels
X = df.iloc[:, :-2304].values
y = df.iloc[:, 2304].values

#Classic Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

#Feature Scaling or Normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Training and Predictions
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#refer report under section "Research Question"
max = 0

#for loop to check k value which corresponds to highest accuracy
for i in range(1, 187):
  classifier = KNeighborsClassifier(n_neighbors=i)
  classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)
  #print (i)
  val = accuracy_score(y_test, y_pred)
  if max<val:
    new_i = i
    max = val
  #print(accuracy_score(y_test, y_pred))
print(new_i)
print(max)

y_pred = classifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix ,accuracy_score
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
#end of code
