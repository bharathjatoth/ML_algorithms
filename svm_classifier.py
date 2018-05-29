from sklearn import linear_model,datasets
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
from sklearn import svm
model = svm.SVC()
model.fit(train_x,train_y)
pred = model.predict(test_x)
#to display the number of
# model = svm.SVC(decision_function_shape='ovo')
# def1 = model.decision_function([[1]])
# print(def1.shape[1])
plt.scatter(test_x,test_y)
plt.plot(test_x,pred)
plt.show()
