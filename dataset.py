from sklearn import linear_model,datasets
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
def data():
  data = datasets.load_diabetes()
  # print(len(data))
  # print(np.newaxis)
  print(data.data[0])
  # print(data['data'][:,1].shape)
  # print(data['data'][:,None,2])
  new = data.data[:,np.newaxis,2]
  # print(len(train_x))
  train_x = new[:-20]
  print(len(train_x))
  test_x = new[-20:]
  train_y = data.target[:-20]
  test_y = data.target[-20:]
  return train_x,train_y,test_x,test_y
 if __name__=="main":
  data()
