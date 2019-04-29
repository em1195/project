import sys
import keras
import numpy as np
from keras import backend as K
from keras import Sequential
from keras.layers import Input, Dense, Lambda, Add, Multiply
from keras.models import Model
from keras import optimizers
import tensorflow as tf
import os
from statistics import mean
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

def gate(X1, X2):
  gate1 = Dense(32,activation='linear',use_bias = True)(X1)
  gate2 = Dense(32, activation='linear')(X2)
  gate3 = Add()([gate1,gate2])
  coeff = Lambda(lambda x: K.sigmoid(x))(gate3)
  return(coeff)

tf.keras.backend.clear_session()
cwd = os.getcwd()

prop = sys.argv[5]
num_layers = int(sys.argv[1])
epochs = int(sys.argv[2])
learning_rate = float(sys.argv[3])
decay_rate = float(sys.argv[4])

feat = np.load(cwd + "/database/ZINC/features/0.npy")
adj = np.load(cwd + "/database/ZINC/adj/0.npy")

conv_feature_dim = 32
readout_dimensions = 512
inputs = [feat,adj]

features_matrix = Input(feat[0].shape)
adj_matrix = Input(adj[0].shape)




X = Dense(32, activation = 'linear', use_bias = True)(features_matrix)
_X = Lambda(lambda x: K.batch_dot(x[0], x[1]))([adj_matrix, X])
_X = Lambda(lambda x: K.relu(x))(_X)

gate1 = Dense(32, activation='linear',use_bias = True)(features_matrix)
gate2 = Dense(32, activation='linear',use_bias= True)(_X)
gate3 = Add()([gate1,gate2])
coeff = Lambda(lambda x: K.sigmoid(x))(gate3)
gated_X = Multiply()([coeff,_X])
gatedX = Multiply()([Lambda(lambda x: 1-x)(coeff), Dense(32,activation=None)(features_matrix)])

_X = Add()([gated_X,gatedX])

conv_output = Lambda(lambda x: K.relu(x))(_X)

for i in range(num_layers-1):
  X = Dense(32, activation = 'linear', use_bias = True)(conv_output)
  _X = Lambda(lambda x: K.batch_dot(x[0], x[1]))([adj_matrix, X])
  _X = Lambda(lambda x: K.relu(x))(_X)
  gate1 = Dense(32,activation='linear',use_bias = True)(conv_output)
  gate2 = Dense(32, activation='linear')(_X)
  gate3 = Add()([gate1,gate2])
  coeff = Lambda(lambda x: K.sigmoid(x))(gate3)
  gated_X = Multiply()([coeff,_X])
  gatedX = Multiply()([Lambda(lambda x: 1-x)(coeff), conv_output])
  conv_output = Add()([gated_X,gatedX])

x2 = Dense(512, activation = 'relu', use_bias = True)(conv_output)
_X = Lambda(lambda x: K.sigmoid(K.sum(K.relu(x), axis=1,keepdims=False)))(x2)
x3 = Dense(512,activation='relu')(_X)
_X = Dense(512, activation='tanh')(x3)
y = Dense(1,activation='linear')(_X)
model = Model(inputs=[features_matrix, adj_matrix], outputs=y)
sgd = optimizers.SGD(lr=learning_rate, decay=decay_rate, momentum=0.1, nesterov=True)
model.compile(optimizer = sgd, loss='mean_absolute_error')
targets = np.load(cwd + "/database/ZINC/"+str(prop)+".npy")
plot_model(model, to_file='GGCN_model.png')
features = np.load(cwd + "/database/ZINC/features/" + str(0) + ".npy")
adjs = np.load(cwd + "/database/ZINC/adj/" + str(0) + ".npy")



for i in range(50):
    print("training GGCN on dataset: ", i+1)
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
    



