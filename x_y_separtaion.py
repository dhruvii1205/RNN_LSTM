import numpy as np

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