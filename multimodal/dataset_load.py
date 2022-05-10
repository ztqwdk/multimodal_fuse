
import torch
import pickle
import random
import os

from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import cv2
import numpy as np

def fetch_datasets(config=""):
    train_dataset = MultiDataset(config, "train")
    test_dataset = MultiDataset(config, "test")

    return train_dataset, test_dataset




class MultiDataset(torch.utils.data.Dataset):
    def __init__(self, config, type):
    
        self.type = type
        self.face_path = "./face_"+type
        self.iris_path = "./iris_"+type
        self.finger_path = "./finger_"+type
        self.to_tensor = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])


    def __len__(self):
        if self.type == "train":
            return 1000
        else:
            return 100

    def __getitem__(self, idx):

        gt_scores = random.randint(1,17)
        folder = str(gt_scores).zfill(2)
        face_filepath = os.path.join(self.face_path, folder)
        iris_filepath = os.path.join(self.iris_path, folder)
        finger_filepath = os.path.join(self.finger_path, folder)
        face_filenames = os.listdir(face_filepath)
        iris_filenames = os.listdir(iris_filepath)
        finger_filenames = os.listdir(finger_filepath)
        
        face_name = random.choice(face_filenames)
        face_img = os.path.join(face_filepath, face_name)
        iris_name = random.choice(iris_filenames)
        iris_img = os.path.join(iris_filepath, iris_name)
        finger_name = random.choice(finger_filenames)
        finger_img = os.path.join(finger_filepath, finger_name)
        
        img_face = Image.open(face_img).convert("RGB")
        img_face = self.to_tensor(img_face)
        
        
        img_iris = Image.open(iris_img).convert("RGB")
        img_iris = self.to_tensor(img_iris)
        
        img_finger = Image.open(finger_img).convert("RGB")
        img_finger = self.to_tensor(img_finger)
        # print(img_finger.shape)


        return img_face, img_iris, img_finger, gt_scores-1


