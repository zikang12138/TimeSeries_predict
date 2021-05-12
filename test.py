import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import statsmodels.api as sm
from pandas import DataFrame , concat
#from sklearn.metrics import mean_absolute_error , mean_squared_error

from numpy import mean , concatenate
from math import sqrt
from pandas import read_csv
#from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Activation,Dropout
from numpy import array , hstack
from tensorflow import keras
import tensorflow as tf
import time
url = 'test.csv'
dataset = pd.read_csv(url)
x_1 = dataset['x1']
y = dataset['y']


x_1 = x_1.values
y = y.values
# plt.figure(figsize=(30, 6))
# plt.plot(x_1[:10064] , label='x1')
# plt.plot(x_2[:10064] , label='x2')

# plt.legend(loc='upper right')
# plt.title("Dataset" ,  fontsize=18)
# plt.xlabel('Time step' ,  fontsize=18)
# plt.ylabel('Values' , fontsize=18)
# plt.legend()
# plt.show()

x_1 = x_1.reshape((len(x_1), 1))
y = y.reshape((len(y), 1))


print ("x_1.shape" , x_1.shape) 
print ("y.shape" , y.shape) 
dataset_stacked = hstack((x_1, y))

print ("dataset_stacked.shape" , dataset_stacked.shape) 
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out-1
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix-1:out_end_ix, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
# choose a number of time steps #change this accordingly
n_steps_in, n_steps_out = 60 , 30 

# covert into input/output
X, y = split_sequences(dataset_stacked, n_steps_in, n_steps_out)
y=np.reshape(y,(y.shape[0],y.shape[1],1))
print ("X.shape" , X.shape) 
print ("y.shape" , y.shape) 
split = 6000
train_X , train_y = X[:split, :] , y[:split, :]
test_X , test_y = X[split:, :] , y[split:, :]

n_features = train_X.shape[2]
#optimizer learning rate
def lstm(x_train, y_train, seq_len,label_len, ep):
        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=(seq_len, 2)))
        model.add(Dropout(0.2))
        model.add(LSTM(100, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(label_len))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer='adam')
        start = time.time()
        model.fit(x_train, y_train, batch_size=64, epochs=ep, validation_split=0.05)
        print('compilation time : ', time.time() - start)
        #model.save(model_save)

lstm(train_X,train_y,n_steps_in,n_steps_out,ep=30)