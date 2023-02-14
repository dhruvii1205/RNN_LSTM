from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import LSTM

# RNN model creation

def rnn_model(X_train):    
  # initializing the RNN
  model = Sequential()

  # adding first RNN layer and dropout regulatization
  model.add(
      SimpleRNN(units = 50,
                activation = "tanh", 
                return_sequences = True, 
                input_shape = (X_train.shape[1],1))
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