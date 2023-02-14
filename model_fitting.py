def model_fitting(model, X_train,y_train, epochs, batch_size):
  # fitting the RNN
  history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)
  return history, model