from sklearn.preprocessing import MinMaxScaler
#Inversing the fit transfrom 
#Actual value from 0-1

def inverse_fit_transform(predict,dataset):
  scaler = MinMaxScaler(feature_range = (0,1))
  fit_scaler = scaler.fit(dataset)
  predict = fit_scaler.inverse_transform(predict) # scaling back from 0-1 to original
  print(predict.shape)

  return predict