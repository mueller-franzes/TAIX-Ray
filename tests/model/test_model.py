import torch 
from cxr.models.model import MST
# from ukacxr.models.resnet import ResNet

input = torch.randn((1,1, 224,224))
model = MST(in_ch=1, out_ch=8)


pred = model(input)
print(pred.shape)
print(pred)
