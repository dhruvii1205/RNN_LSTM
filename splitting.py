# Splitting data
# 30% test and 70% train

def train_test_splitting(data):
  len_data = len(data)
  split_ratio = 0.7           # %70 train + %30 validation
  length_train = round(len_data * split_ratio)  
  length_validation = len_data - length_train
  print("Data length :", len_data)
  print("Train data length :", length_train)
  print("Validation data lenth :", length_validation)
  return length_train, length_validation
