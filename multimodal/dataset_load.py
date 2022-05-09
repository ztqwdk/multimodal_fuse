import torchaudio
import torch
import pickle


from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
import cv2

def fetch_datasets(config=""):
    train_dataset = MultiDataset(config, "train")
    test_dataset = MultiDataset(config, "test")

    return train_dataset, test_dataset




class MultiDataset(torch.utils.data.Dataset):
    def __init__(self, config, type):
    
        self.type = type
        self.config = config
        with open('/home/multimodal/'+ type +'_data.pkl', 'rb') as f:
            data = pickle.load(f)
        self.data = data


    def __len__(self):
        return self.data['gt_scores'].shape[0]

    def __getitem__(self, idx):

        input1 = self.data['input1'][idx]
        input2 = self.data['input2'][idx]
        input3 = self.data['input3'][idx]

        gt_scores = self.data['gt_scores'][idx]


        return input1, input2, input3, gt_scores


