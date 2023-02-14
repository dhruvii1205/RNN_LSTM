from sklearn.preprocessing import MinMaxScaler
# Normalising data
# (0-1)

def preprocessing_data(dataset):
  scaler = MinMaxScaler(feature_range = (0,1)) #setting range from 0 to 1 

  # scaling dataset
  dataset_scaled = scaler.fit_transform(dataset)

  return dataset_scaled
