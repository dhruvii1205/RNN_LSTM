from keras.models import Model
from keras.models import load_model

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