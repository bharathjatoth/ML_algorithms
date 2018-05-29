from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(train_x,train_y)
pred = model.predict(test_x)
plt.scatter(test_x,test_y)
plt.plot(test_x,pred)
plt.show()
