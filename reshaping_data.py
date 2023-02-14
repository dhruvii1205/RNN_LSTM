import numpy as np
# Reshaping data 
# (1182,) to (1182,1) making it a 2d array

def reshape(data_reshape):   #reshapeing the data
  dataset = data_reshape.Open.values
  dataset = np.reshape(dataset, (-1,1))
  print(dataset.shape)
  return dataset