import torch
import os
import numpy as np
import random

import timm
from timm.data import create_dataset
from timm.data.dataset import ImageDataset
from timm.data.loader import create_loader
from timm.data.transforms_factory import create_transform

from transformers import AdamW
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
    
    train_dataset,  test_dataset = fetch_datasets(config)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, num_workers=0, shuffle=True)
    model = MultiModel(10000).to(device)
    # model = model
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50000, eta_min=5e-5)
    
    best_val_acc = 0
    best_epoch = 0
    for epoch in range(500):
        training_loss = 0
        train_right = 0
        train_false = 0
        
        model.train()
        # train_tqdm = tqdm(train_loader)
        # for step, batch in enumerate(train_tqdm):
        for step, batch in enumerate(train_loader):
            input1, input2, input3, label = batch
            output = model(input1, input2, input3)
            loss = F.cross_entropy(output, label)
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            predict = torch.argmax(output, 1)
            
            train_right_count = predict==label
            train_false_count = predict!=label
            train_right += sum(train_right_count)
            train_false += sum(train_false_count)
            training_loss += loss.item()
            # train_tqdm.set_postfix(loss=loss.item())
        
        if epoch % 10 == 0:
            print(f'epoch为{epoch}, 训练loss为{training_loss/len(train_loader)}')
            print(f'训练集准确率为{train_right/(train_right+train_false)}')
            val_right = 0
            val_false = 0
            
            model.eval()
            with torch.no_grad():
                val_tqdm = tqdm(val_loader)
                for step, batch in enumerate(val_tqdm):
                # for step, batch in enumerate(val_loader):
                    input, label = batch
                    output = model(input)
                    loss = F.cross_entropy(output, label)

                    predict = torch.argmax(output, 1)
                    val_right_count = predict==label
                    val_false_count = predict!=label
                    val_right += sum(val_right_count)
                    val_false += sum(val_false_count)
                val_acc = val_right/(val_right + val_false)
                    
            if val_acc > best_val_acc:
                best_epoch = epoch
                best_val_acc = val_acc
                torch.save(model.state_dict(), "./models/best.pt")
            print(f'测试集准确率为{val_acc}')
            print(f'目前测试集最高准确率为{best_val_acc}，epoch为{best_epoch}')
    print(f'测试集最高准确率为{best_val_acc}，epoch为{best_epoch}')
            