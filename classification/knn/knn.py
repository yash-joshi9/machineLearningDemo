import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


dataset = pd.read_csv("./Social_Network_Ads.csv")


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)


knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train.ravel())

y_pred = knn.predict(X_test)

x = accuracy_score(y_test, y_pred)

print(x)