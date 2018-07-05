#visualizations in deeper way this will show all the visualizations in the extreme level
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

log_path = r'C:\Users\jatoth.kumar\PycharmProjects\Tensorflow\tmp1\logs1'
mnist = input_data.read_data_sets('mnist_data/',one_hot=True)

#parameters
learning_rate = 0.01
epochs = 10
batch_size = 100
n_hidden1 = 128
n_hidden2 = 256

x = tf.placeholder("float",[None,784])
y = tf.placeholder("float",[None,10])
w = {"w1":tf.Variable(tf.random_normal([784,n_hidden1])),"w2":tf.Variable(tf.random_normal([n_hidden1,n_hidden2])),"out":tf.Variable(tf.random_normal([n_hidden2,10]))}
b = {"b1":tf.Variable(tf.random_normal([n_hidden1])),"b2":tf.Variable(tf.random_normal([n_hidden2])),"out_b":tf.Variable(tf.random_normal([10]))}
#model build
def neural(x,w,b):
    layer_1 = tf.add(tf.matmul(x,w['w1']),b['b1'])
    layer_1 = tf.nn.relu(layer_1)
    tf.summary.histogram('layer1',layer_1)
    layer_2 = tf.add(tf.matmul(layer_1,w['w2']),b['b2'])
    layer_2 = tf.nn.relu(layer_2)
    tf.summary.histogram('layer_2',layer_2)
    return tf.add(tf.matmul(layer_2,w['out']),b['out_b'])
init = tf.global_variables_initializer()
pred = neural(x,w,b)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
with tf.name_scope('SGD'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    grads = tf.gradients(loss,tf.trainable_variables())
    grads = list(zip(grads,tf.trainable_variables()))
tf.summary.scalar("loss",loss)
accuracy = tf.equal(tf.argmax(y,1),tf.argmax(pred,1))
acc = tf.reduce_mean(tf.cast(accuracy,tf.float32))
tf.summary.scalar('accuracy',acc)
for i,j in grads:
    tf.summary.histogram(j.name,i)
merged_summ = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(log_path,graph=tf.get_default_graph())
    avg_cost = 0
    for i in range(epochs):
        for j in range(int(mnist.train.num_examples/batch_size)):
            train_x,train_y = mnist.train.next_batch(batch_size)
            _,c,summar = sess.run([optimizer,loss,merged_summ],feed_dict={x:train_x,y:train_y})
            avg_cost+=c
            summary_writer.add_summary(summar)
        print(avg_cost)
    print(acc.eval({x:mnist.test.images,y:mnist.test.labels}))
