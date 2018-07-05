import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist_data/",one_hot=True)

learning_rate = 0.01
epochs = 20
batch_size = 100
log_path = r'C:\Users\jatoth.kumar\PycharmProjects\Tensorflow\tmp1\logs1'

x = tf.placeholder(tf.float32, [None,784], name="input")
y = tf.placeholder(tf.float32, [None,10], name="output")

W = tf.Variable(tf.zeros([784,10]),name='weights')
b = tf.Variable(tf.zeros([10]),name="bias")

with tf.name_scope('Model'):
    pred = tf.nn.softmax(tf.matmul(x,W)+b)
with tf.name_scope('loss'):
    cost = tf.reduce_mean(-tf.reduce_mean(y*tf.log(pred)))
with tf.name_scope('SGD'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
with tf.name_scope('accuracy'):
    acc = tf.equal(tf.argmax(y,1),tf.argmax(pred,1))
    acc = tf.reduce_mean(tf.cast(acc,tf.float32))

init = tf.global_variables_initializer()

tf.summary.scalar("loss",cost)
tf.summary.scalar("accuracy",acc)
merged_summary = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(log_path,graph=tf.get_default_graph())
    for epoch in range(epochs):
        cost_avg = 0
        for j in range(int(mnist.train.num_examples/batch_size)):
            train_x,train_y = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer,cost],feed_dict={x:train_x,y:train_y})
            cost_avg += c
    print("optimization finished")
    print(acc.eval({x:mnist.test.images,y:mnist.test.labels}))
