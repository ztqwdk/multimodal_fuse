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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser()
parser.add_argument("--type", default='face', type=str, help='data type')
args = parser.parse_args()

train_path = "./"+args.type+"_train"
val_path = "./"+args.type+"_val"

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

    
    file = torch.load("./temp/"+args.type+"_features.pt")
    features = file["feature"].to(device)
    labels = file["label"].to(device)
    
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0, in_chans=3, checkpoint_path="").to(device)

    pre_dict = torch.load("./models/"+args.type+"_best.pt")
    model.load_state_dict(pre_dict, strict=False)

    val_right = 0
    val_false = 0
    
    dataset_val = create_dataset("", root=val_path, transform=create_transform(224) )


    try:
        # only works if gpu present on machine
        val_loader = create_loader(dataset_val, (3, 224, 224), 1)
    except:
        val_loader = create_loader(dataset_val, (3, 224, 224), 1, use_prefetcher=False)

    
    model.eval()
    with torch.no_grad():

        for step, batch in enumerate(val_loader):
            input, label = batch
            output = model(input)
            output = F.normalize(output, dim=1)
            
            logits = (features @ output.T)
            # print(logits)
            
            predict = labels[torch.argmax(logits, 0).item()]
            if logits[torch.argmax(logits, 0).item()].item() >= 0.8:
                print(f'测试者编号为{int(predict)+1}')
            else:
                print(f'测试图片不在库中')
            # print(logits[torch.argmax(logits, 0).item()].item())



            
