# RNN_LSTM
stock price predictions with rnn and lstm model.

### How to set-up the project?
* Firstly, download the repository. Refer this link if stuck: https://blog.hubspot.com/website/download-from-github
* After downloading the repository, download the libraries mentioned in requirements.txt file, if not.
* Open the stock_price_prediction.py file in any editor and run the code. 
* As the script takes dynamic arguements as input write **python3 stock_price_prediction_final.py(file name with extension) -c Tesla.csv (data file name wih extension) -m rnn (model name which you waant to use in lower-case) -mode train(train/predict)**
* If you want to try LSTM model then insert lstm with -m

### What each funcionality is doing?

* Datetime object and feature selection
```
def convert_datetime_object(data, length_train_validation):
  train_test_data = data[:length_train_validation].iloc[:,:2] 
  train_test_data['Date'] = pd.to_datetime(train_test_data['Date'])  # converting to date time object
  return train_test_data    #returning the data
```

* By using this function, we will get first two columns and will work on them for the rest of the code which is opening stock price and date. 
* To predict the next opening stock price we will be using these two columns. 

### Splitting data
```
def train_test_splitting(data):
  len_data = len(data)
  split_ratio = 0.7           # %70 train + %30 validation
  length_train = round(len_data * split_ratio)  
  length_validation = len_data - length_train
  print("Data length :", len_data)
  print("Train data length :", length_train)
  print("Validation data lenth :", length_validation)
  return length_train, length_validation

```
* We will be splitting the data to train and validation in which 70% data will be given to training and rest to validation

### Preprocessing data
```
def preprocessing_data(dataset):
  scaler = MinMaxScaler(feature_range = (0,1)) #setting range from 0 to 1 

  # scaling dataset
  dataset_scaled = scaler.fit_transform(dataset)

  return dataset_scaled
```
* By doing data pre-processing we will get the whole data between the given range, here it is between 0 and 1. 
* We do data pre-processing because it will speed up the time and normalise the data

### X and Y saperation
```
def create_x_y(data_length, dataset_scaled):
  X_data = []
  y_data = []

  time_step = 1

  for i in range(time_step, data_length):
      X_data.append(dataset_scaled[i-time_step:i,0])
      y_data.append(dataset_scaled[i,0])
    
```
* X and y separation, with x data we will predict y value. 
* We will get a list of X and Y will be their expected output. 

### Inversing MinMaxScaler
```
def inverse_fit_transform(predict,dataset):
  scaler = MinMaxScaler(feature_range = (0,1))
  fit_scaler = scaler.fit(dataset)
  predict = fit_scaler.inverse_transform(predict) # scaling back from 0-1 to original
  print(predict.shape)
```
* To get the actual value we will have to inverse the fit transform function. 
* We have **minverse_transform** to do that.

### Predicting the values
```
def prediction(X_train, model, model_name, weights):
  if model_name == 'rnn':
    model.load_weights(weights)
    predicted = model.predict(X_train)
    return predicted
  else:
    model.load_weights(weights)
    predict = model.predict(X_train)
    return predict
```
* If the model we use is RNN then first half will be executed and second half for LSTM 
* The weights value will also be given to load the saved weight. 

### What Train or predict will do? 
* By giving train as a parameter, the model will be trained and the weights will be saved to a specific path.
* BY giving predict as a parameter, we will use the saved weights by using **load_weight** function and predict the values for train and test dataset along with plotting the output for both the case. 
* We will also print the **MSE (Mean Sqaured Error)** to understand which model is giving better result. 
