#import relevant libraries
import pandas
import numpy
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

df = pandas.read_csv("C:/Users/Aisha Aijaz Ahmad/Desktop/fer2018.csv")#load dataset
df.head()

#split our dataset into its attributes and labels
X = df.iloc[:, :-2304].values
y = df.iloc[:, 2304].values
validation_size = 0.4
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y,
test_size=validation_size, random_state=seed)

# Split-out validation dataset
array = df.values
target_index = df.columns.get_loc("emotion") # obtain the 'status' location
Y = array[:,target_index]
features = df.drop(["emotion"], axis=1) #drop the 'status' target
X = features #features.values

# refer report under section "Random Forests"
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=200,
oob_score = True, random_state = 42)
random_forest.fit(X_train, Y_train)
Y_pred_rf = random_forest.predict(X_validation)
random_forest.score(X_train, Y_train)
rf_accur = accuracy_score(Y_validation, Y_pred_rf)
print(confusion_matrix(Y_validation, Y_pred_rf))
print(classification_report(Y_validation, Y_pred_rf))
rf_accur

#end of code
