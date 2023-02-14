import matplotlib.pyplot as plt
# Data Plotting

def plotting(y_pred, y_train,label_x,label_y, title, figure_path):

  # visualisation
  plt.figure(figsize = (16,10))
  plt.plot(y_pred, color = "b", label = label_x)
  plt.plot(y_train, color = "g", label = label_y)
  plt.xlabel("Days")
  plt.ylabel("Opening price")
  plt.title(title)
  plt.legend()
  plt.show()
  plt.savefig(figure_path)