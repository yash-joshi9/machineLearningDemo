import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score


dataset = pd.read_csv("./Social_Network_Ads.csv")


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.fit_transform(X_test)




regressor = LogisticRegression(random_state=0).fit(X_train, y_train)


# c = regressor.predict(sc.transform([[30, 87000]]))
y_pred = regressor.predict(X_test)


# print(np.concatenate( (y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)) , 1))


c = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)


print(f"accuracy {accuracy}")