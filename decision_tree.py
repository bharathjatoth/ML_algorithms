#decision Tree refer to the following link for more information
#http://scikit-learn.org/stable/modules/tree.html#tree
# to split into different groups tree uses  various techniques like Gini, Information Gain, Chi-square, entropy.
from sklearn import linear_model,datasets
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np
from sklearn import tree
import graphviz
model = tree.DecisionTreeClassifier(criterion="gini",splitter="best")
x2 = model.fit(train_x,train_y)
#graphviz is for the vizualisation of the tree
d_data = tree.export_graphviz(x2)
graph = graphviz.Source(d_data)
graph.save()
pred = model.predict(test_x)
plt.scatter(train_x,train_y)
plt.plot(test_x,pred)
plt.show()
