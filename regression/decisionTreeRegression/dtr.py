import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor



dataset = pd.read_csv("./Position_Salaries.csv")


X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values

regressor = DecisionTreeRegressor(random_state=0)

regressor.fit(X, y)


c = regressor.predict([[6.5]])


import matplotlib.pyplot as plt 



x_grid = np.arange(min(X), max(X), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)

plt.scatter(X, y, color="red")

plt.plot(x_grid, regressor.predict(x_grid), color="blue")
plt.show()



plt.scatter(X, y, color="red")

plt.plot(X, regressor.predict(X), color="blue")
plt.show()
