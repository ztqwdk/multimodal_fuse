import torch
import os
import numpy as np
import random

import timm
from timm.data import create_dataset
from timm.data.dataset import ImageDataset
from timm.data.loader import create_loader
from timm.data.transforms_factory import create_transform

import torch.nn.functional as F
from tqdm.auto import tqdm
from model import MultiModel
from dataset_load import fetch_datasets

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# train_path = "/home/data_iris/train"

def set_random_seeds(seed):
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__=="__main__":
    seed = 42

    set_random_seeds(seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: ",device)
    train_dataset,  test_dataset = fetch_datasets()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, num_workers=0, shuffle=True)
    model = MultiModel(0).to(device)

    # pre_dict = torch.load("./models/CASIA4_efficientnet_b0/best.pt")
    pre_dict = torch.load("./models/best.pt")
    model.load_state_dict(pre_dict, strict=False)
    # model = model

    
    ori_right = 0
    ori_false = 0
    
    features = torch.Tensor()
    labels = torch.Tensor()
    model.eval()
    with torch.no_grad():
        train_tqdm = tqdm(train_loader)
        for step, batch in enumerate(train_tqdm):
            input1, input2, input3, label = batch
            input1 = input1.to(device)
            input2 = input2.to(device)
            input3 = input3.to(device)
            label = label.to(device)
            output = model(input1, input2, input3)
            output = F.normalize(output, dim=1)
            features = torch.cat((features, output.cpu()), 0)
            labels = torch.cat((labels, label.cpu()), 0)
    print(features.shape)
    print(labels.shape)
    torch.save({"feature":features,"label":labels}, "./temp/features.pt")
            
        

            
