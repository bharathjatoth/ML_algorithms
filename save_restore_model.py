#script written on 5th July 2018 by --//\\%%
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist_data/",one_hot=True)
# mnist = ("mnist_data/")
learning_rate = 0.01
batch_size = 100
n_layers_1 = 32
n_layers_2 = 64
n_input = 784
n_classes = 10
path = r'C:\Users\jatoth.kumar\PycharmProjects\Tensorflow\tmp1\model_new.ckpt'
epochs = 10
x = tf.placeholder("float",[None,n_input])
y = tf.placeholder("float",[None,n_classes])

def neural_net(x,weights,bias):
    layer_1 = tf.add(tf.matmul(x,weights['w1']),bias['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1,weights['w2']),bias['b2'])
    layer_2 = tf.nn.relu(layer_2)
    out = tf.add(tf.matmul(layer_2,weights['out']),bias['out'])
    return out

weights = {"w1":tf.Variable(tf.random_normal([n_input,n_layers_1])),"w2":tf.Variable(tf.random_normal([n_layers_1,n_layers_2])),
           "out":tf.Variable(tf.random_normal([n_layers_2,n_classes]))}
bias = {"b1":tf.Variable(tf.random_normal([n_layers_1])),"b2":tf.Variable(tf.random_normal([n_layers_2])),
           "out":tf.Variable(tf.random_normal([n_classes]))}
#creates a model
model = neural_net(x,weights,bias)
#cost and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

#saving a model to the location specified
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess,path)
    for i in range(epochs):
        cost_avg = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for j in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
    print("First Optimization Finished!")

    correct__pred = tf.equal(tf.argmax(model,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct__pred,"float"))
    save_path = saver.save(sess,path)
    print(save_path)

#restoring the model which is trained above and saved
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess,path)
    for i in range(epochs):
        avg_cost = 0
        x1 = int(mnist.train.num_examples/batch_size)
        for j in range(x1):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
            avg_cost+=c
        print(avg_cost)
    correct_pred = tf.equal(tf.argmax(y,1),tf.argmax(model,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,"float"))
    print(accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
