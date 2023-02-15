# -*- coding: utf-8 -*-
"""stock_price_prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AZe4E9rNs8vBogo5idEefaJsi5wsK9Lp

# Importing Libraries
"""

import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import argparse
from keras.models import Model
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from utils import *
from models import *

parser = argparse.ArgumentParser(description='choosing correct parameters for csv type(-c) model(-m) and mode(-mode)')
parser.add_argument('-c','--csv_file',type=str, required=True, help='enter the csv file name with extension')
parser.add_argument('-m', '--model_name', type=str, required=  True, help='choose the model (RNN/LSTM)')
parser.add_argument('-mode', '--mode', type=str, required=True, help='choose the mode (Train/ Test)')

args = parser.parse_args()


# Function calls 

def data_reading(filename, model_name, mode, weights_path_rnn, weights_path_lstm):
  data = pd.read_csv(filename) # reading the csv file
  data.info() # to understand the coulumns and their type

  train_length, validation_length = train_test_splitting(data)    # train_test_split data 

  train_data = convert_datetime_object(data, train_length)   # getting train data
  validation_data = convert_datetime_object(data, validation_length)   # getting validation data

  dataset_train = reshape(train_data)
  dataset_train_scaled = preprocessing_data(dataset_train)  # normalising data between 0 and 1
  
  X_train, y_train =  create_x_y(train_length, dataset_train_scaled)

  if model_name == 'rnn':
    if mode == 'train':
      epochs = 50
      batch_size = 32

      # RNN model
      
      model = rnn_model(X_train.shape)   

      history, model= model_fitting(model, X_train, y_train, epochs, batch_size)
      
      model.save_weights(weights_path_rnn) 
      

    if mode == 'predict':
      if weights_path_rnn:
        model = rnn_model(X_train.shape)  

        reshaped_vaidation_data = reshape(validation_data)
        scaled_dataset_validation = preprocessing_data(reshaped_vaidation_data)
  
        X_test, y_test = create_x_y(validation_length, scaled_dataset_validation)   #getting test data and reshaping them

        y_pred_of_test = prediction(X_test, model, model_name, weights_path_rnn)   #predicting the output

        y_pred_of_test = inverse_fit_transform(y_pred_of_test, dataset_train)   #inverse_transform to get the actual value 
        y_test = inverse_fit_transform(y_test, dataset_train)

        plotting(y_pred_of_test, y_test,'y_pred_of_test','y_test','test data predictions with RNN', 'rnn_plot.jpg')
        print('Mean sqaure error of y_test and predicted value for RNN is: ', mean_squared_error(y_test, y_pred_of_test))
      
      else:
        print('You might have chose predict mode before train mode. Train the model first before predicting values.')
  
  if model_name == 'lstm':
    if mode == 'train':
      epochs = 10
      batch_size = 32

      # LSTM model
      model_lstm = lstm_model(X_train.shape) 
      history, model_lstm= model_fitting(model_lstm, X_train, y_train, epochs, batch_size)
      model_lstm.save_weights(weights_path_lstm)
  
    if mode == 'predict':
      if weights_path_lstm:
        model_lstm = lstm_model(X_train.shape) 
        
        reshaped_vaidation_data = reshape(validation_data)
        scaled_dataset_validation = preprocessing_data(reshaped_vaidation_data)
  
        X_test, y_test = create_x_y(validation_length, scaled_dataset_validation)   #getting test data and reshaping them

        y_pred_of_test = prediction(X_test, model_lstm, model_name, weights_path_lstm)  #predicting the output

        y_pred_of_test = inverse_fit_transform(y_pred_of_test, dataset_train)   #inverse_transform to get the actual value 
        y_test = inverse_fit_transform(y_test, dataset_train)

        plotting(y_pred_of_test, y_test,'y_pred_of_test','y_test','Test data predictions with LSTM', 'lstm_figure.png')
        print('Mean sqaure error of y_train and predicted value for LSTM is: ', mean_squared_error(y_test, y_pred_of_test))
      
      else:
        print('You might have chose predict mode before train mode. Train the model first before predicting values.')
        
# Giving the path of the csv file

def reading_file_csv(csv_file, model_name, mode):
  weights_path_rnn = 'my_weights.model'
  weights_path_lstm = 'my_weights1.model'
  data_reading(csv_file, model_name, mode, weights_path_rnn, weights_path_lstm)
  print('done')


if __name__ == '__main__':
  reading_file_csv(args.csv_file, args.model_name, args.mode)