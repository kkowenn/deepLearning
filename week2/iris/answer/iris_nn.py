# iris_nn.py
# Iris classification Keras 2.1.4 TensorFlow 1.4.0

import numpy as np
import keras as K

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main():
  print("Iris dataset using Keras/TensorFlow ")

  print("Loading Iris data into memory \n")
  data_file = ".\\Data\\iris_data.txt"
  train_x = np.loadtxt(data_file, usecols=[0,1,2,3],
    delimiter=",",  skiprows=0, dtype=np.float32)
  train_y = np.loadtxt(data_file, usecols=[4,5,6],
    delimiter=",", skiprows=0, dtype=np.float32)

  np.random.seed(4)
  model = K.models.Sequential()
  model.add(K.layers.Dense(units=7, input_dim=4,
    activation='tanh'))
  model.add(K.layers.Dense(units=3, activation='softmax'))
  model.compile(loss='categorical_crossentropy',
    optimizer='sgd', metrics=['accuracy'])

  print("Starting training \n")
  h = model.fit(train_x, train_y, batch_size=1,
    epochs=12, verbose=1)  # 1 = very chatty
  print("\nTraining finished \n")

  eval = model.evaluate(train_x, train_y, verbose=0)
  print("Evaluation: loss = %0.6f  accuracy = %0.2f%% \n" \
    % (eval[0], eval[1]*100) )

  np.set_printoptions(precision=4)
  unknown = np.array([[6.1, 3.1, 5.1, 1.1]],
    dtype=np.float32)
  predicted = model.predict(unknown)
  print("Using model to predict species for features: ")
  print(unknown)
  print("\nPredicted species is: ")
  print(predicted)

if __name__=="__main__":
  main()
