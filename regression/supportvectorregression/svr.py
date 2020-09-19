import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import  SVR



dataset = pd.read_csv("./Position_Salaries.csv")

X = dataset.iloc[:,1: -1].values
y = dataset.iloc[:, -1:].values

sc_x = StandardScaler()
sc_y = StandardScaler()


X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)


regressor = SVR(kernel = 'rbf')

regressor.fit(X, y)

pred = regressor.predict(sc_x.transform([[6.5]]))

pred = sc_y.inverse_transform(pred)



# plt.scatter(sc_x.inverse_transform(X), sc_y.inverse_transform(y), color="red")
# plt.plot(sc_x.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color="blue")
# plt.show()


# higher resolution

X_grid = np.arange(min(sc_x.inverse_transform(X)), max(sc_x.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(sc_x.inverse_transform(X), sc_y.inverse_transform(y), color="red")
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_x.transform(X_grid))), color="blue")
plt.show()











