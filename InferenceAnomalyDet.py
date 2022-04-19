from keras.models import load_model
from reader import read
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

model = load_model('C:\\Users\\slmcy\\Downloads\\Cloud_model3_best.h5')

#Read data.
data_path = "datasets/vulnbank_train.txt"
data_array = read(data_path)

#Divide data into train and test.
data_train,data_test = train_test_split(data_array,test_size=0.2,random_state=0)
print("Number of train datas:",data_train.shape[0])
print("Number of test datas:",data_test.shape[0])

X_train = data_train.reshape(data_train.shape[0], 1, data_train.shape[1])
print("Training data shape:", X_train.shape)
X_test = data_test.reshape(data_test.shape[0], 1, data_test.shape[1])
print("Test data shape:", X_test.shape)

# calculate the loss on the test set
X_pred = model.predict(X_test)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
data_test = pd.DataFrame(data_test)
X_pred = pd.DataFrame(X_pred, columns=data_test.columns)
X_pred.index = data_test.index

scored = pd.DataFrame(index=data_test.index)
Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
scored['Threshold'] = 10
scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
scored.head()

sum = 0
for i in range(0,len(scored)):
    sum += np.asarray(scored)[i][0]

print("Sum of losses:",sum)
mean_benign=sum/3906
print("Mean of benign losses:",mean_benign)



count=0
for i in range(0,len(scored)):
  if(np.asarray(scored)[i,2]==False):
    count=count+1
print("Number of TN",count)

#-----------------------------------------------------------------------------#

# plot the training losses
# fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
# ax.plot(model['loss'], 'b', label='Train', linewidth=2)
# ax.plot(model['val_loss'], 'r', label='Validation', linewidth=2)
# ax.set_title('Model loss', fontsize=16)
# ax.set_ylabel('Loss (mae)')
# ax.set_xlabel('Epoch')
# ax.legend(loc='upper right')
# plt.show()

# data_array_anom = read("datasets/vulnbank_anomaly.txt")
# X_test_anom = data_array_anom.reshape(data_array_anom.shape[0], 1, data_array_anom.shape[1])
#
# start = time.time()
# X_pred = model.predict(X_test_anom)
# middle = time.time()
# X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
# data_array_anom = pd.DataFrame(data_array_anom)
# X_pred = pd.DataFrame(X_pred, columns=data_array_anom.columns)
# X_pred.index = data_array_anom.index
#
# scored = pd.DataFrame(index=data_array_anom.index)
# Xtest = X_test_anom.reshape(X_test_anom.shape[0], X_test_anom.shape[2])
# scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
# scored['Threshold'] = 10
# scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
# scored.head()
# end = time.time()
# print(end-start,"pred time")
#
# sum = 0
# for i in range(0,len(scored)):
#     sum += np.asarray(scored)[i][0]
#
# mean_malicious=sum/1097
# print("Mean of malicious losses:",mean_malicious)
#
# count=0
# for i in range(0,len(scored)):
#   if(np.asarray(scored)[i,2]==True):
#     count=count+1
# print("Number of TP",count)


