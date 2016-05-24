import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0] 
def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


data = np.genfromtxt('iris.data', delimiter=",")
np.random.shuffle(data)
x_data = data[:,0:4].astype('f4')
y_data = one_hot(data[:,4].astype(int), 3)

print y_data

print "\nSome samples..."
for i in range(20):
    print x_data[i], " -> ", y_data[i]
print

x = tf.placeholder("float", [None, 4])
y_ = tf.placeholder("float", [None, 3])

neurons_in_hidden_layer = 5

weight_hidden_layer = tf.Variable(np.float32(np.random.rand(4, neurons_in_hidden_layer)) * 0.1)
bias_hidden_layer = tf.Variable(np.float32(np.random.rand(neurons_in_hidden_layer)) * 0.1)

weight = tf.Variable(np.float32(np.random.rand(neurons_in_hidden_layer, 3)) * 0.1)
bias = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

layer_output = tf.sigmoid(tf.matmul(x, weight_hidden_layer) + bias_hidden_layer)
y = tf.nn.softmax(tf.matmul(layer_output, weight) + bias)

cross_entropy = tf.reduce_sum(tf.square(y_ - y))
#cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20
list_of_errors = []

for step in xrange(1000):
    for jj in xrange(len(x_data) / batch_size):
        batch_xs = x_data[jj*batch_size : jj*batch_size+batch_size]
        batch_ys = y_data[jj*batch_size : jj*batch_size+batch_size]

        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
        if step % 50 == 0:
            error = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
            print "Iteration #:", step, "Error: ", error
            list_of_errors.append(error)
            result = sess.run(y, feed_dict={x: batch_xs})
            for bias, r in zip(batch_ys, result):
                print bias, "-->", r
            print "----------------------------------------------------------------------------------"
plt.plot(list_of_errors)
plt.show()
