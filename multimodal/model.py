import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

device = torch.device('cuda')

class MultiModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.model1 = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, in_chans=3, checkpoint_path="")
        self.model2 = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, in_chans=3, checkpoint_path="")
        self.model3 = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, in_chans=3, checkpoint_path="")
        
        if (num_classes!=0):
            self.linear = nn.Linear(1280, self.num_classes)
    
    def forward(self, input1, input2, input3):

        input1 = self.model1(input1)
        input2 = self.model2(input2)
        input3 = self.model3(input3)
        
        # output = torch.cat((input1, input2, input3), 0)
        
        output = input1 + input2 + input3

        if (self.num_classes):
            output = self.linear(output)
        

        return output



if __name__=="__main__":
    pass
