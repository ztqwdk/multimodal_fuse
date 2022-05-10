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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser()
parser.add_argument("--type", default='face', type=str, help='data type')
args = parser.parse_args()

train_path = "./"+args.type+"_train"
val_path = "./"+args.type+"_test"


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
    dataset_train = create_dataset("", root=train_path, transform=create_transform(224))


    try:
        # only works if gpu present on machine
        train_loader = create_loader(dataset_train, (3, 224, 224), 1)
    except:
        train_loader = create_loader(dataset_train, (3, 224, 224), 1, use_prefetcher=False)
    
    
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0, in_chans=3, checkpoint_path="").to(device)

    # pre_dict = torch.load("./models/CASIA4_efficientnet_b0/best.pt")
    pre_dict = torch.load("./models/"+args.type+"_best.pt")
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
    torch.save({"feature":features,"label":labels}, "./temp/"+args.type+"_features.pt")
            
        

            
