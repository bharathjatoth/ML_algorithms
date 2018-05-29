#K-Means Unsuper
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3,random_state=0)
model.fit(train_x,train_y)
pred = model.predict(test_x)
plt.scatter(test_x,test_y)
plt.plot(test_x,pred)
plt.show()
