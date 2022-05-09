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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
train_path = "/home/CASIA4-Thousand/train"
val_path = "/home/CASIA4-Thousand/val"
ori_path = "/home/CASIA4-Thousand/ori"

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
    '''
    dataset_ori = create_dataset("", root=ori_path, transform=create_transform(64))


    try:
        # only works if gpu present on machine
        ori_loader = create_loader(dataset_ori, (3, 64, 64), 1)
    except:
        ori_loader = create_loader(dataset_ori, (3, 64, 64), 1, use_prefetcher=False)
    '''
    dataset_train = create_dataset("", root=train_path, transform=create_transform(64))


    try:
        # only works if gpu present on machine
        train_loader = create_loader(dataset_train, (3, 64, 64), 8)
    except:
        train_loader = create_loader(dataset_train, (3, 64, 64), 8, use_prefetcher=False)
    
    
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0, in_chans=3, checkpoint_path="").to(device)

    # pre_dict = torch.load("./models/CASIA4_efficientnet_b0/best.pt")
    pre_dict = torch.load("./models/data_iris/best.pt")
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
            input, label = batch
            output = model(input)
            output = F.normalize(output, dim=1)
            features = torch.cat((features, output.cpu()), 0)
            labels = torch.cat((labels, label.cpu()), 0)
    print(features.shape)
    print(labels.shape)
    torch.save({"feature":features,"label":labels}, "./temp/features.pt")
            
        

            
