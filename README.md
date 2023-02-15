# RNN_LSTM
stock price predictions with rnn and lstm model.

### How to setup the program?
* Firstly, you need to download this repository. 
* Open the repository on your local computer and run the main.py file. 
* After that, as this script is using argparse() method we can give dynamic input in the terminal
* The syntax is: *python3 main.py -c Tesla.csv(data file along with extension) -m rnn (model type either rnn or lstm) -mode train(either train or predict)*
* If you want to train or predict using lstm model give lstm as -m input.
* utils.py file contains all the functions such as pre-processing, plotting_data etc.
* models.py file contains the functions to call lstm or rnn model and model fit function. 

### What output should we expect?
* If we give train as an input to -mode then the given model will be train and model weights will be saved at specific destination.
* If we give predict as an input to -mode then the saved weights will be loaded and we will predict the test dataset accodingly. 
* We will also be able to see the graph of actual output of test data and predicted output. 
* Also, **MSE(Mean Sqauared Error)** will be calculated to understand which model is working better.

### Working of function

```
def create_x_y(data_length, dataset_scaled):
  X_data = []
  y_data = []

  time_step = 1

  for i in range(time_step, data_length):
      X_data.append(dataset_scaled[i-time_step:i,0])
      y_data.append(dataset_scaled[i,0])

  X_data, y_data = np.array(X_data), np.array(y_data)

  print("Shape of X_data before reshape :",X_data.shape)
  print("Shape of y_data before reshape :",y_data.shape)

  X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1],1)) 
  y_data = np.reshape(y_data, (y_data.shape[0],1))

  print("Shape of X_train after reshape :",X_data.shape)
  print("Shape of y_train after reshape :",y_data.shape)

  return X_data, y_data

```
* By calling this function we can make a list of X_data and y_data.
* In which X_data will contain the input of the model and y_data will be the output we should expect.
* time_step is a variable which is taking the number of input values for which we will predict y_data. 
* If time_step value is 50 then after reshape we will get (1823,50,1) 

#### Data pre-processing

```
def preprocessing_data(dataset):
  scaler = MinMaxScaler(feature_range = (0,1)) 

  # scaling dataset
  dataset_scaled = scaler.fit_transform(dataset)

  return dataset_scaled
```

* By using this function we will be able to convert our data from 0 to 1.
* We normalise data for faster calculations. 

```
def inverse_fit_transform(predict,dataset):
  scaler = MinMaxScaler(feature_range = (0,1))
  fit_scaler = scaler.fit(dataset)
  predict = fit_scaler.inverse_transform(predict) 
  print(predict.shape)

  return predict
```
* This process reverse the calculations of preprocessing_data function. 

#### Feature selection

```
def convert_datetime_object(data, length_train_validation):
  train_test_data = data[:length_train_validation].iloc[:,:2] 
  train_test_data['Date'] = pd.to_datetime(train_test_data['Date'])  
  return train_test_data  
```

* By using this function, we are choosing two columns which are date and opening price of stock. 
* As we want to predict the opening price as ouput we are just choosing two columns. 
* Also, we are converting date to datetime object. 

#### Train test splitting
```
def train_test_splitting(data):
  len_data = len(data)
  split_ratio = 0.7           
  length_train = round(len_data * split_ratio)  
  length_validation = len_data - length_train
  print("Data length :", len_data)
  print("Train data length :", length_train)
  print("Validation data lenth :", length_validation)
  return length_train, length_validation
```
* By using this function we can get 70% data as train data and 30% data as validation. 

#### Prediction

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
* Predict function will predict the test data by unloading the weights and using them to predict the output.
