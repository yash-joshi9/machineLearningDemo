import pandas as pd
import numpy as np
from  sklearn.ensemble import RandomForestRegressor


dataset = pd.read_csv("./Position_Salaries.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values



regessor = RandomForestRegressor(n_estimators=10, random_state=0)


regessor.fit(X, y)

c = regessor.predict([[6.5]])

