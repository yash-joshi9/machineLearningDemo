import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv("./Position_Salaries.csv")

X = dataset.iloc[:,1: -1].values
y = dataset.iloc[:, -1:].values

# simple linear regressor
lin_regressor = LinearRegression()
lin_regressor.fit(X, y)


poly_regressor = PolynomialFeatures(degree=4)
X_poly = poly_regressor.fit_transform(X)

# polynomial linear regressor

lin_regressor_2 = LinearRegression()
lin_regressor_2.fit(X_poly, y)



# plt.scatter(X, y)
# plt.plot(X, lin_regressor_2.predict(X_poly), color="red")
# plt.show()



# X_grid = np.arange(min(X), max(X), 0.1)
# X_grid = X_grid.reshape(len(X_grid), 1)

# X_poly_grid = poly_regressor.fit_transform(X_grid)

# plt.plot(X_grid, lin_regressor_2.predict(X_poly_grid), color="red")
# plt.show()


y_pred_linear = lin_regressor.predict([[6.5]])

y_pred_poly = lin_regressor_2.predict(poly_regressor.fit_transform([[6.5]]))

print(y_pred_linear)
print(y_pred_poly)
