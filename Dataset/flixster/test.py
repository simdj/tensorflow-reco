import numpy as np
import json
try:
    import tensorflow as tf
except:
    pass
train_filename = 'train_100_1.csv'
train_file = open(train_filename,'r')

movie_vector_len = int(train_filename.split("_").pop().split(".")[0])*1000


# def get_train_data_format(line):
#     train_json = json.loads(line)
# #     print train_json['my']
# #     print train_json['social']
#     x = np.zeros(n_input)
    
#     for key in train_json['my']:
#         x[int(key)]=train_json['my'][key]
    
#     for key in train_json['social']:
#         x[n_input/2+int(key)]=train_json['social'][key]
#     # view view_indicater
#     y = (x[0:movie_vector_len]>0).astype(float)
#     return x, y


# # concatentate my_vector and social_vector
# n_input = movie_vector_len+movie_vector_len
# n_output = movie_vector_len

# n_hidden = 100
# corruption_level = 0.3



# row_cnt=0

# line = train_file.next()
# data_x , data_y = get_train_data_format(line)
# row_cnt+=1

# for line in train_file:
#     # x, y = get_train_data_format(line)

#     # data_x = np.vstack([data_x, x])
#     # data_y = np.vstack([data_y, y])
#     row_cnt+=1
#     # if row_cnt>=10000:
#     #     break

# train_x = data_x[:int(row_cnt*0.8),:]
# train_y = data_y[:int(row_cnt*0.8),:]
# test_x = data_x[int(row_cnt*0.8):,:]
# test_y = data_y[int(row_cnt*0.8):,:]
import time
start =time.time()

row_cnt=31
ll=np.array(list(range(row_cnt)))*10
print row_cnt
for i in xrange(row_cnt/30+1):
	print i
	if i*30!=min((i+1)*30,row_cnt):
		print ll[i*30:min((i+1)*30, row_cnt)]


print 
for i,v in zip(ll.argsort()[-10:],ll[ll.argsort()[-10:]]):
	print (i,v),
time.sleep(2.3)
print ''
print '{:.0f} zz'.format(time.time()-start)