from sklearn import linear_model,datasets
import matplotlib.pyplot as plt
from dataset import data

from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
train_x,train_y,test_x,test_y = data()

#logistic Regression model which predicts the outcome as 0/1, True/False
#to improve this we can use regularisation, non-linear model, feature scaling
model = linear_model.LogisticRegression(C=100,penalty='l2')
model.fit(train_x,train_y)
pred = model.predict(test_x)
plt.scatter(train_x,train_y)
plt.plot(test_x,pred)
plt.show()
print(pred)
