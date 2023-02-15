import matplotlib.pyplot as plt
from keras.models import Model
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


# Adding data to x and y 
# x -> input & y -> output

def create_x_y(data_length, dataset_scaled):
  X_data = []
  y_data = []

  time_step = 1

  for i in range(time_step, data_length):
      X_data.append(dataset_scaled[i-time_step:i,0])
      y_data.append(dataset_scaled[i,0])
    
  # convert list to array

  X_data, y_data = np.array(X_data), np.array(y_data)

  print("Shape of X_data before reshape :",X_data.shape)
  print("Shape of y_data before reshape :",y_data.shape)

  # making the data 3d so that when we change the timestamp we can work fluently.
  #if we choose timestamp = 50 then out array will look like (1182,50,1)

  X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1],1)) 
  y_data = np.reshape(y_data, (y_data.shape[0],1))

  print("Shape of X_train after reshape :",X_data.shape)
  print("Shape of y_train after reshape :",y_data.shape)

  return X_data, y_data


#Inversing the fit transfrom 
#Actual value from 0-1

def inverse_fit_transform(predict,dataset):
  scaler = MinMaxScaler(feature_range = (0,1))
  fit_scaler = scaler.fit(dataset)
  predict = fit_scaler.inverse_transform(predict) # scaling back from 0-1 to original
  print(predict.shape)

  return predict

# Converting to datetime object and Feature selection

def convert_datetime_object(data, length_train_validation):
  train_test_data = data[:length_train_validation].iloc[:,:2] 
  train_test_data['Date'] = pd.to_datetime(train_test_data['Date'])  # converting to date time object
  return train_test_data    #returning the data

# Splitting data
# 30% test and 70% train

def train_test_splitting(data):
  len_data = len(data)
  split_ratio = 0.7           # %70 train + %30 validation
  length_train = round(len_data * split_ratio)  
  length_validation = len_data - length_train
  print("Data length :", len_data)
  print("Train data length :", length_train)
  print("Validation data lenth :", length_validation)
  return length_train, length_validation


# Reshaping data 
# (1182,) to (1182,1) making it a 2d array

def reshape(data_reshape):   #reshapeing the data
  dataset = data_reshape.Open.values
  dataset = np.reshape(dataset, (-1,1))
  print(dataset.shape)
  return dataset

# Normalising data
# (0-1)

def preprocessing_data(dataset):
  scaler = MinMaxScaler(feature_range = (0,1)) #setting range from 0 to 1 

  # scaling dataset
  dataset_scaled = scaler.fit_transform(dataset)

  return dataset_scaled

# Prediction
def prediction(X_train, model, model_name, weights):
  if model_name == 'rnn':
    model.load_weights(weights)
    predicted = model.predict(X_train)
    return predicted
  else:
    model.load_weights(weights)
    predict = model.predict(X_train)
    return predict

# Data Plotting

def plotting(y_pred, y_train,label_x,label_y, title, figure_path):

  # visualisation
  plt.figure(figsize = (16,10))
  plt.plot(y_pred, color = "b", label = label_x)
  plt.plot(y_train, color = "g", label = label_y)
  plt.xlabel("Days")
  plt.ylabel("Opening price")
  plt.title(title)
  plt.legend()
  plt.savefig(figure_path)
  plt.show()
  