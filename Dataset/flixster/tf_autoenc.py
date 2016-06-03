import numpy as np
import json
import time
try:
    import tensorflow as tf
except:
    pass

train_filename = 'train_100_1.csv'
movie_vector_len = int(train_filename.split("_").pop().split(".")[0])*1000


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
    y = (x[0:movie_vector_len]>0).astype(float)
    return x, y


# concatentate my_vector and social_vector
n_input = movie_vector_len+movie_vector_len
n_output = movie_vector_len

n_hidden = 100

#################
# preprocessing #
#################
start =time.time()
row_cnt=0

train_file = open(train_filename,'r')
line = train_file.next()
data_x , data_y = get_train_data_format(line)
row_cnt+=1

for line in train_file:
    x, y = get_train_data_format(line)

    data_x = np.vstack([data_x, x])
    data_y = np.vstack([data_y, y])
    row_cnt+=1
    if row_cnt>=10000:
        break
train_file.close()

train_size = int(row_cnt*0.8)
train_x = data_x[:train_size,:]
train_y = data_y[:train_size,:]
test_x = data_x[train_size:,:]
test_y = data_y[train_size:,:]
end = time.time()
print '[Preprocessing done]'
print '\tElapsed time:{:.0f}'.format(end-start)
print '\tData size:', row_cnt, 'Train size', train_size, 'Test size',row_cnt-train_size




###################
# TF model config #
###################

# create node for input data
X = tf.placeholder("float", [None, n_input], name='X')
# create node for output data
Y = tf.placeholder("float", [None, n_input/2], name='Y')

# create nodes for hidden variables
W_init_max = 4 * np.sqrt(6. / (n_input + n_hidden))
W_init = tf.random_uniform(shape=[n_input, n_hidden], minval=-W_init_max, maxval=W_init_max)
W_prime_init = tf.random_uniform(shape=[n_hidden, n_output], minval=-W_init_max, maxval=W_init_max)

W = tf.Variable(W_init, name='W')
b = tf.Variable(tf.zeros([n_hidden]), name='b')

W_prime = tf.Variable(W_prime_init, name='W_prime')
b_prime = tf.Variable(tf.zeros([n_output]), name='b_prime')


def model(X, W, b, W_prime, b_prime):
    H = tf.nn.sigmoid(tf.matmul(X, W) + b)  # hidden state
    Z = tf.nn.sigmoid(tf.matmul(H, W_prime) + b_prime)  # //reconstructed input
    return Z

# build model graph
Z = model(X, W, b, W_prime, b_prime)

# create cost function
observed_Z = tf.select(tf.greater(Y,0), Z, tf.zeros_like(Z))
cost = tf.reduce_sum(tf.pow(Y - observed_Z, 2))  # minimize squared error
train_op = tf.train.GradientDescentOptimizer(0.02).minimize(cost)  # construct an optimizer

# you need to initialize all variables
init = tf.initialize_all_variables()

# Launch the graph in a session
sess = tf.Session()
sess.run(init)

batch_size = 30

print '[TF configuration done]'


#########################################
############ learning ###############
#########################################
start = time.time()
for i in range(101):
    # SGD!!
    for j in xrange(train_size/batch_size+1):
        start = j*batch_size
        end = min((j+1)*batch_size,train_size)
        if start!=end:
            sess.run(train_op, feed_dict = { 
                X: train_x[start:end,:],
                Y: train_y[start:end,:]
                })

    if i%10==0:
        print i, 'Cost',sess.run(cost, feed_dict = { X:test_x, Y:test_y})
    #     print test_y[0:1]
end = time.time()
print '[TF learning done]'
print '\tElapsed time:{:.0f}'.format(end-start)



###############
## inference ##
###############
query = 56
estimated_y = sess.run(Z, feed_dict = { X:test_x[query].reshape(1,n_input), Y:test_y[query].reshape(1,n_output), })
observed_estimated_y = sess.run(observed_Z, feed_dict = { X:test_x[query].reshape(1,n_input), Y:test_y[query].reshape(1,n_output), })


print ' Top 10 prob'
print estimated_y[0].argsort()[-10:][::-1]
# for i,v in zip(estimated_y[0].argsort()[-10:],estimated_y[0][estimated_y[0].argsort()[-10:]]):
#     print (i,v),
print '\n My prob'
my_list = []
for i,v in enumerate(test_x[query][:n_output]):
    if v>0:
        print (i,v),
        my_list.append((i,v))


print '\n Social prob'
social_list = []
for i,v in enumerate(test_x[query][n_output:]):
    if v>0:
        # print (i,v),
        social_list.append((i,v))

for (movie,rating) in my_list:
    for (s_movie,s_count) in social_list:
        if movie==s_movie:
            print (s_movie,s_count),


print '\n Observed estimated Y'
for i,v in enumerate(observed_estimated_y[0]):
    if v>0:
        print (i,v),


# print '\n Estimated Y'
# for i,v in enumerate(estimated_y[0]):
#     if v>0.9:
#         print (i,v),

print ''
for border in [0.1, 0.3, 0.5, 0.9, 0.99]:
    print ' Less than', border,'->', len(estimated_y[0][estimated_y[0]<border])