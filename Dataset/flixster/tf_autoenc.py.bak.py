import tensorflow as tf
import numpy as np
import json

def get_train_data_format(line):
    train_json = json.loads(line)
#     print train_json['my']
#     print train_json['social']
    x = np.zeros(n_input)
    
    for key in train_json['my']:
        x[int(key)]=train_json['my'][key]
    
    for key in train_json['social']:
        x[n_input/2+int(key)]=train_json['social'][key]
    # view view_indicater
    y = (x[0:1000]>0).astype(float)
    return x, y


# concatentate my_vector and social_vector
n_input = 1000+1000
n_output = 1000

n_hidden = 50
corruption_level = 0.3



train_file = open('train_100_1.csv','r')
row_cnt=0

line = train_file.next()
train_x , train_y = get_train_data_format(line)
row_cnt+=1

for line in train_file:
    x, y = get_train_data_format(line)

    train_x = np.vstack([train_x, x])
    train_y = np.vstack([train_y, y])
    row_cnt+=1
    if row_cnt>=10:
        break
print (train_x).shape






# create node for input data
X = tf.placeholder("float", [None, n_input], name='X')

Y = tf.placeholder("float", [None, n_input/2], name='Y')
# create node for corruption mask
# mask = tf.placeholder("float", [None, n_input], name='mask')



# create nodes for hidden variables
W_init_max = 4 * np.sqrt(6. / (n_input + n_hidden))
W_init = tf.random_uniform(shape=[n_input, n_hidden],
                           minval=-W_init_max,
                           maxval=W_init_max)

W = tf.Variable(W_init, name='W')
b = tf.Variable(tf.zeros([n_hidden]), name='b')

# W_prime = tf.transpose(W)  # tied weights between encoder and decoder
W_prime_init = tf.random_uniform(shape=[n_hidden, n_output],
                           minval=-W_init_max,
                           maxval=W_init_max)
W_prime = tf.Variable(W_prime_init, name='W_prime')
b_prime = tf.Variable(tf.zeros([n_output]), name='b_prime')


def model(X,  W, b, W_prime, b_prime):
    # tilde_X = mask * X  # corrupted X
    # Y = tf.nn.sigmoid(tf.matmul(tilde_X, W) + b)  # hidden state

    H = tf.nn.sigmoid(tf.matmul(X, W) + b)  # hidden state
    Z = tf.nn.sigmoid(tf.matmul(H, W_prime) + b_prime)  # //reconstructed input
    return Z

# build model graph
Z = model(X, W, b, W_prime, b_prime)


# a = tf.constant([[1, 2, 3], [4, 5, 6]])
# mask = tf.constant([[True, False, True], [True, False, False]])
# print sess.run(tf.select(mask, a, tf.zeros_like(a)))

# create cost function
observed_Z = tf.select(tf.greater(Y,0), Z, tf.zeros_like(Z))
cost = tf.reduce_sum(tf.pow(Y - observed_Z, 2))  # minimize squared error
train_op = tf.train.GradientDescentOptimizer(0.02).minimize(cost)  # construct an optimizer

# load MNIST data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    # for i in range(100):
    #     for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
    #         input_ = trX[start:end]
    #         mask_np = np.random.binomial(1, 1 - corruption_level, input_.shape)
    #         sess.run(train_op, feed_dict={X: input_, mask: mask_np})

    #     mask_np = np.random.binomial(1, 1 - corruption_level, teX.shape)
    #     print(i, sess.run(cost, feed_dict={X: teX, mask: mask_np}))
print 'end'