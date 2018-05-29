#K-Means Unsuper
from dataset import data
from sklearn import linear_model,datasets
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
train_x,train_y,test_x,test_y = data()
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3,random_state=0)
model.fit(train_x,train_y)
pred = model.predict(test_x)
plt.scatter(test_x,test_y)
plt.plot(test_x,pred)
plt.show()
