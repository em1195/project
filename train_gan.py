import sys
import keras
import numpy as np
from keras import backend as K
from keras import Sequential
from keras.layers import Input, Dense, Lambda, Flatten, Add, Multiply
from keras.models import Model
from keras import optimizers
import tensorflow as tf
import os
from statistics import mean
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

def attention(X,A,num_heads):
  
  attn_list = []
  for i in range(num_heads):
    tX = Lambda(lambda x: K.permute_dimensions(x, (0,2,1)))(X)
    tX = Dense(32, activation='linear')(tX)
    X1 = Lambda(lambda x: K.batch_dot(x[0],x[1]))([X,tX])
    _A = Lambda(lambda x: K.batch_dot(x[0], x[1]))([A, X1])
    _A = Lambda(lambda x: K.tanh(x))(_A)
    
    attn_list.append(_A)
  
  
  _A = Add()(attn_list)
  _A = Lambda(lambda x: x/num_heads)(_A)
  
  return(_A)

tf.keras.backend.clear_session()
cwd = os.getcwd()

prop = sys.argv[5]
num_layers = int(sys.argv[1])
epochs = int(sys.argv[2])
learning_rate = float(sys.argv[3])
decay_rate = float(sys.argv[4])
num_heads = 4
heads_list = []

feat = np.load(cwd + "/database/ZINC/features/0.npy")
adj = np.load(cwd + "/database/ZINC/adj/0.npy")

conv_feature_dim = 32
readout_dimensions = 512
inputs = [feat,adj]

features_matrix = Input(feat[0].shape)
adj_matrix = Input(adj[0].shape)

_X = Lambda(lambda x: x)(features_matrix)
for i in range(num_heads):
  X = Dense(32, activation = 'linear', use_bias = True)(_X) # X*weights
  attn = attention(X, adj_matrix,num_heads) #returns _A to be timesed by _X
  h = Multiply()([attn,X])
  heads_list.append(h)
  
_X = Add()(heads_list)  
_X = Lambda(lambda x: K.relu(x/num_heads))(_X)
#_X = Lambda(lambda x: K.relu(x))(_X) #computes recified linear
_X = Add()([_X,X]) #computed skipped connection
conv_output = Lambda(lambda x: K.relu(x))(_X) #compute recified linear


for i in range(num_layers-1):
  _X = Lambda(lambda x: x)(conv_output)
  for i in range(num_heads):
    X = Dense(32, activation = 'linear', use_bias = True)(_X) # X*weights
    attn = attention(X, adj_matrix,num_heads) #returns _A to be timesed by _X
    h = Multiply()([attn,X])
    heads_list.append(h)
  
  _X = Add()(heads_list)  
  _X = Lambda(lambda x: K.relu(x/num_heads))(_X)
  _X = Add()([_X,X]) #computed skipped connection
  conv_output = Lambda(lambda x: K.relu(x))(_X) #compute recified linear


x2 = Dense(512, activation = 'relu', use_bias = True)(conv_output)
_X = Lambda(lambda x: K.sigmoid(K.sum(K.relu(x), axis=1,keepdims=False)))(x2)
x3 = Dense(512,activation='relu')(_X)
_X = Dense(512, activation='tanh')(x3)
y = Dense(1,activation='linear')(_X)


model = Model(inputs=[features_matrix,adj_matrix], outputs=y)
sgd = optimizers.SGD(lr=learning_rate, decay=decay_rate, momentum=0.1, nesterov=True)
model.compile(optimizer = sgd, loss='mean_absolute_error')
targets = np.load(cwd + "/database/ZINC/"+str(prop)+".npy")
features = np.load(cwd + "/database/ZINC/features/" + str(0) + ".npy")
adjs = np.load(cwd + "/database/ZINC/adj/" + str(0) + ".npy")


for i in range(50):
    print("training GCN on dataset: ", i+1)
    features = np.load(cwd + "/database/ZINC/features/" + str(i) + ".npy")
    adjs = np.load(cwd + "/database/ZINC/adj/" + str(i) + ".npy")
    data = [features,adjs]
    target = targets[i*10000:(i+1)*10000]
    
    history = model.fit(data, target,batch_size=16, epochs = epochs, validation_split=0.1)
    
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
    
