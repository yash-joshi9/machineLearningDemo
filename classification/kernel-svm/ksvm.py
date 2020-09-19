import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC



dataset = pd.read_csv("./Social_Network_Ads.csv")


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)


classifier = SVC(gamma="auto", kernel="rbf")


classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

x =  accuracy_score(y_pred, y_test)

print(x)