#KNN is computationally expensive
# Variables should be normalized else higher range variables can bias it
# Works on pre-processing stage more before going for kNN like outlier, noise removal
from dataset import data
from sklearn import linear_model,datasets
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
train_x,train_y,test_x,test_y = data()

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(train_x,train_y)
pred = model.predict(test_x)
plt.scatter(test_x,test_y)
plt.plot(test_x,pred)
plt.show()
