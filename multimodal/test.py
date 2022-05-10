import torch
import os
import numpy as np
import random
import argparse

import timm
from timm.data import create_dataset
from timm.data.dataset import ImageDataset
from timm.data.loader import create_loader
from timm.data.transforms_factory import create_transform

import torch.nn.functional as F
from tqdm.auto import tqdm
from torchvision import transforms
from PIL import Image
from model import MultiModel
from dataset_load import fetch_datasets

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



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

    
    file = torch.load("./temp/features.pt")
    features = file["feature"].to(device)
    labels = file["label"].to(device)
    
    model = MultiModel(0).to(device)

    pre_dict = torch.load("./models/best.pt")
    model.load_state_dict(pre_dict, strict=False)

    val_right = 0
    val_false = 0
    
    to_tensor = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])
    
    img_face = Image.open("face.jpg").convert("RGB")
    img_face = to_tensor(img_face).unsqueeze(0)
    
    
    img_iris = Image.open("iris.jpg").convert("RGB")
    img_iris = to_tensor(img_iris).unsqueeze(0)
    
    img_finger = Image.open("finger.jpg").convert("RGB")
    img_finger = to_tensor(img_finger).unsqueeze(0)

    
    model.eval()
    with torch.no_grad():

        input1 = img_face.to(device)
        input2 = img_iris.to(device)
        input3 = img_finger.to(device)
        output = model(input1, input2, input3)
        output = F.normalize(output, dim=1)
        
        logits = (features @ output.T)
        # print(logits)
        
        predict = labels[torch.argmax(logits, 0).item()]
        if logits[torch.argmax(logits, 0).item()].item() >= 0.8:
            print(f'测试者编号为{int(predict)+1}')
        else:
            print(f'测试组合不在库中，或输入模态来自不同人')
        print(logits[torch.argmax(logits, 0).item()].item())



            
