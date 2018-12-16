import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load dataset
dataset = pd.read_csv("C:/Users/Aisha Aijaz Ahmad/Desktop/coursework data mining/use
these/CORR_TOP_10.csv")

#knn
array = dataset.values
x = array[:, 0:70]
y = array[:, 70]

#split the data
x_train, x_test,y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state
= 42, stratify =y)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#refer report section under "PCA"
#make instance of the model
pca = PCA(.95)
pca.fit(x_train)

#to see how many components were selected
total_no_comp = pca.n_components_
print(total_no_comp)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

#applying kNN
knn = KNeighborsClassifier(n_neighbors =3)
knn.fit(x_train, y_train)
y_pred= knn.predict(x_test)
print(knn.score(x_test, y_test))
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))

#compute and plot the training and testing accuracy scores for a variety of different
neighbor values as shown in the report
neighbors = np.arange(1,50)
train_acc = np.empty(len(neighbors))
test_acc = np.empty(len(neighbors))

#run code to get accuracies per k
for i, k in enumerate(neighbors):
  knn= KNeighborsClassifier(n_neighbors = k)
  knn.fit(x_train, y_train)
  train_acc[i] = knn.score(x_train, y_train)
  test_acc[i] = knn.score(x_test, y_test)

#plot the values
plt.plot(neighbors, train_acc, label= "train set accuracy")
plt.plot(neighbors, test_acc, label= "test set accuracy")
plt.legend()
plt.xlabel("No. of neighbors")
plt.ylabel("Accuracy")
plt.title("kNN accuracy with varying number of neighbors")
plt.show()

#end of code
