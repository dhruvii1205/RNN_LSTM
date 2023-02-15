from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import LSTM

# RNN model creation

def rnn_model(X_train_shape):    
  # initializing the RNN
  model = Sequential()

  # adding first RNN layer and dropout regulatization
  model.add(
      SimpleRNN(units = 50,
                activation = "tanh", 
                return_sequences = True, 
                input_shape = (X_train_shape[1],1))
              )

  model.add(Dropout(0.2))

  model.add(
      SimpleRNN(units = 50, 
                activation = "tanh", 
                return_sequences = True)
              )

  model.add(Dropout(0.2))

  model.add(
      SimpleRNN(units = 50, 
                activation = "tanh", 
                return_sequences = True)
              )

  model.add(Dropout(0.2))

  model.add(SimpleRNN(units = 50))

  model.add(Dropout(0.2))

  # output layer
  model.add(Dense(units = 1))
  #compiler
  model.compile(
      optimizer = "adam", 
      loss = "mean_squared_error",
      metrics = ["accuracy"])
  
  return model


# LSTM model 

def lstm_model(X_train_shape):
  model_lstm = Sequential()
  model_lstm.add(
      LSTM(64,return_sequences=True,input_shape = (X_train_shape[1],1))) #64 lstm neuron block
  model_lstm.add(
      LSTM(64, return_sequences= False))
  model_lstm.add(Dense(32))
  model_lstm.add(Dense(1))
  model_lstm.compile(loss = "mean_squared_error", optimizer = "adam", metrics = ["accuracy"])

  return model_lstm

def model_fitting(model, X_train,y_train, epochs, batch_size):
  # fitting the RNN
  history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)
  return history, model