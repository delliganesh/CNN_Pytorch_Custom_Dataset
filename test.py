import os
import torch
import numpy as np
from  dataloder import custom_dataset
from torch.utils.data import DataLoader
from parameters import MODEL_NAME, test_data_path
from sklearn.metrics import classification_report, confusion_matrix

test_dataset = custom_dataset(test_data_path)

test_data = DataLoader(test_dataset,batch_size=1, shuffle=True)
classifier = torch.load('models/{}.pt'.format(MODEL_NAME)).eval()
predicted ,actual = [], []
for _,data in enumerate(test_data,0):
    test_x, test_y = data
    pred = classifier(test_x)
    y_pred = np.argmax(pred.data)
    actual.append(chr(test_y[0]+65))
    predicted.append(chr(y_pred+65))
target_names = [chr(i+65) for i in range(10)]
print("***************confusion_matrix****************************\n")
print(confusion_matrix(actual, predicted, labels=target_names))
print("\n**************Classification Report************************")
print(classification_report(actual, predicted, target_names=target_names))


