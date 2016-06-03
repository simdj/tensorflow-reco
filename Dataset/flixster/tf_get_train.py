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