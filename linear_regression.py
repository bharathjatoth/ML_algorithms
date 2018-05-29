from dataset import data
from sklearn import linear_model,datasets
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np

reg  = linear_model.LinearRegression()
reg.fit(train_x,train_y)
pred = reg.predict(test_x)

print(mean_squared_error(pred,test_y),r2_score(pred,test_y))

plt.scatter(test_x,test_y)
plt.plot(test_x,pred)

plt.show()
