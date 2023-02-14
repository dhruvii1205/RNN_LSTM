
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import LSTM

# LSTM model 

def lstm_model(X_train):
  model_lstm = Sequential()
  model_lstm.add(
      LSTM(64,return_sequences=True,input_shape = (X_train.shape[1],1))) #64 lstm neuron block
  model_lstm.add(
      LSTM(64, return_sequences= False))
  model_lstm.add(Dense(32))
  model_lstm.add(Dense(1))
  model_lstm.compile(loss = "mean_squared_error", optimizer = "adam", metrics = ["accuracy"])

  return model_lstm