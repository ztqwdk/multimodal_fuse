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
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0)
    model = MultiModel(0).to(device)
    
    file = torch.load("./temp/features.pt")
    features = file["feature"].to(device)
    labels = file["label"].to(device)

    pre_dict = torch.load("./models/best.pt")
    model.load_state_dict(pre_dict, strict=False)

    val_right = 0
    val_false = 0
    
    # test = [0]*1000
    
    model.eval()
    with torch.no_grad():
        test_tqdm = tqdm(test_loader)
        for step, batch in enumerate(test_tqdm):
            input1,input2,input3, label = batch
            # test[label.item()] += 1
            input1 = input1.to(device)
            input2 = input2.to(device)
            input3 = input3.to(device)
            label = label.to(device)
            output = model(input1,input2,input3)
            output = F.normalize(output, dim=1)
            
            logits = (features @ output.T)
            predict = labels[torch.argmax(logits, 0).item()]
            # print(predict)
            val_right_count = predict==label
            val_false_count = predict!=label
            val_right += sum(val_right_count).item()
            val_false += sum(val_false_count).item()
            val_acc = val_right/(val_right + val_false)
            test_tqdm.set_postfix(acc=val_acc)
            
        val_acc = val_right/(val_right + val_false)
        print(f'测试集准确率为{val_acc}')
        # print(test)


            
