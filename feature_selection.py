import pandas as pd
# Converting to datetime object and Feature selection

def convert_datetime_object(data, length_train_validation):
  train_test_data = data[:length_train_validation].iloc[:,:2] 
  train_test_data['Date'] = pd.to_datetime(train_test_data['Date'])  # converting to date time object
  return train_test_data    #returning the data
