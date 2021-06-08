import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import cv2

class custom_dataset:
    def __init__(self, train_path):
        images, label = [],[]
        folders=os.listdir(train_path)
        for foldername in folders:
            folder_path = os.path.join(train_path,foldername)
            for filename in os.listdir(folder_path):
                imagepath = os.path.join(folder_path,filename)
                try:
                    img = cv2.imread(imagepath)
                    images.append(np.array(img))
                    label.append(ord(foldername)-65)
                except:
                    print("Invalid Image")
        data = [(x,y) for x,y in zip(images,label)]
        self.data = data

    def __len__(self):
        return(len(self.data))

    def __getitem__(self,index):
        img = self.data[index][0]
        img_tensor = torch.from_numpy(img).view(3, 28, 28).float()
        label = self.data[index][1]
        return img_tensor,label

