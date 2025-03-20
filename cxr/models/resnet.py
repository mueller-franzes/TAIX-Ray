
import torch 
import torch.nn as nn 
import torchvision.models as models
import  torch.optim.lr_scheduler as lr_scheduler


from .base_model import BasicClassifier, BasicRegression

def _get_resnet_torch(model):
    return {
        18: models.resnet18, 34: models.resnet34, 50: models.resnet50, 101: models.resnet101, 152: models.resnet152
    }.get(model) 

class ResNet(BasicClassifier):
    def __init__(
        self,
        in_ch=1, 
        out_ch=1, 
        task="multilabel",
        spatial_dims=2,
        model = 18,
        pretrained=True,
        optimizer_kwargs={'lr':1e-5, 'weight_decay':1e-2},
        # lr_scheduler= lr_scheduler.LinearLR, 
        # lr_scheduler_kwargs={'start_factor':1e-3, 'total_iters':10000},
        **kwargs
    ):
        super().__init__(in_ch, out_ch, task, spatial_dims, 
                    optimizer_kwargs=optimizer_kwargs, 
                    # lr_scheduler=lr_scheduler, 
                    # lr_scheduler_kwargs=lr_scheduler_kwargs, 
                    **kwargs)
        Model = _get_resnet_torch(model)
        weights='DEFAULT' if pretrained else None 
        self.model = Model(weights=weights)
        
        emb_ch = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.linear = nn.Linear(emb_ch, out_ch)

    def forward(self, x_in, **kwargs):
        x = x_in.repeat(1, 3, 1, 1) # Gray to RGB
        hidden = self.model(x)
        logits = self.linear(hidden)
        return logits



class ResNetRegression(BasicRegression):
    def __init__(
        self, 
        in_ch=1,
        out_ch=1, 
        task="ordinal",
        spatial_dims=2,
        model = 34,
        pretrained=True,
        optimizer_kwargs={'lr':1e-5, 'weight_decay':1e-2},
        # lr_scheduler= lr_scheduler.LinearLR, 
        # lr_scheduler_kwargs={'start_factor':1e-3, 'total_iters':10000},
        **kwargs
    ):
        super().__init__(in_ch, out_ch, task, spatial_dims, 
                    optimizer_kwargs=optimizer_kwargs, 
                    # lr_scheduler=lr_scheduler, 
                    # lr_scheduler_kwargs=lr_scheduler_kwargs, 
                    **kwargs)
        Model = _get_resnet_torch(model)
        weights='DEFAULT' if pretrained else None 
        self.model = Model(weights=weights)
        
        emb_ch = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.linear = nn.Linear(emb_ch, out_ch)

    def forward(self, x_in):
        x = x_in.repeat(1, 3, 1, 1) # Gray to RGB
        hidden = self.model(x)
        logits = self.linear(hidden)
        return logits


    