import torch 
from cxr.models import MST, ResNet

input = torch.randn((1,1, 224,224))
# model = MST(in_ch=1, out_ch=8)
model = ResNet(in_ch=1, out_ch=8)


pred = model(input)
print(pred.shape)
print(pred)
