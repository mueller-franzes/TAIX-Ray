
import torch 
import torch.nn as nn 
import torchvision.models as models
import  torch.optim.lr_scheduler as lr_scheduler
from .base_model import BasicClassifier, BasicRegression
from transformers import AutoModel, PreTrainedModel, PretrainedConfig
from torch.utils.checkpoint import checkpoint
    


class MSTRegression(BasicRegression):
    def __init__(
        self,
        in_ch, 
        out_ch, 
        task="ordinal",
        spatial_dims=2,
        optimizer_kwargs={'lr':1e-6, 'weight_decay':1e-2},
        # lr_scheduler= lr_scheduler.LinearLR, 
        # lr_scheduler_kwargs={'start_factor':1e-3, 'total_iters':1000},
        **kwargs
    ):
        super().__init__(in_ch, out_ch, task, spatial_dims, 
                         optimizer_kwargs=optimizer_kwargs, 
                        #  lr_scheduler=lr_scheduler, 
                        #  lr_scheduler_kwargs=lr_scheduler_kwargs, 
                         **kwargs
                        )

        self.model = torch.hub.load('facebookresearch/dinov2', f'dinov2_vits14')
        emb_ch = self.model.num_features 
        self.linear = nn.Linear(emb_ch, out_ch)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1) # Gray to RGB
        # x = self.model(x) #  -> [B, out] 
        x = checkpoint(self.model, x.requires_grad_())
        x = self.linear(x)
        return x
    




class MST(BasicClassifier):
    def __init__(
        self,
        in_ch,
        out_ch, 
        task="binary",
        spatial_dims=2,
        optimizer_kwargs={'lr':1e-6, 'weight_decay':1e-2},
        # optimizer_kwargs={'lr':5e-4, 'weight_decay':1e-2},
        # lr_scheduler= lr_scheduler.LinearLR, 
        # lr_scheduler_kwargs={'start_factor':1e-3, 'total_iters':10000},
        **kwargs
    ):
        super().__init__(in_ch, out_ch, task, spatial_dims,
                         optimizer_kwargs=optimizer_kwargs, 
                        #  lr_scheduler=lr_scheduler, 
                        #  lr_scheduler_kwargs=lr_scheduler_kwargs, 
                         **kwargs
                        )

        self.model = torch.hub.load('facebookresearch/dinov2', f'dinov2_vits14')
        emb_ch = self.model.num_features 
        # self.model = AutoModel.from_pretrained("microsoft/rad-dino")
        # emb_ch = 768 
        self.linear = nn.Linear(emb_ch, out_ch)



    def forward(self, x):
        x = x.repeat(1, 3, 1, 1) # Gray to RGB
        # x = self.model(x) 
        x = checkpoint(self.model, x.requires_grad_())
        # x = x.pooler_output
        x = self.linear(x)
        return x
    


    def forward_attention(self, x_in):
        with torch.no_grad():
            x = self.model.prepare_tokens_with_masks(x_in.repeat(1, 3, 1, 1))
            for blk in self.model.blocks[:-1]:
                x = blk(x)

            # Compute normal output 
            pred =  self.forward_mask(self.model.blocks[-1], x, None)
            
            # Compute last block with masked attention 
            token_relevance = []
            for token_i in range(1, x.shape[1]):
                pred_i = self.forward_mask(self.model.blocks[-1], x, token_i)
                rel_change = (pred.sigmoid() - pred_i.sigmoid()).abs() 
                token_relevance.append(rel_change)

        token_relevance = torch.stack(token_relevance, dim=1)


        return pred, token_relevance
    
    def forward_mask(self, block, x, mask_i):
        """
        Recompute the class token representation with the contribution of one patch token removed.
        """
        x = self.run_block(block, x, mask_i)
        x_norm = self.model.norm(x)
        x_norm_clstoken = x_norm[:, 0]
        pred = self.linear(x_norm_clstoken)
        return pred
    

    def run_block(self, block, x, mask_i):
        x = x + block.ls1(self.attn(block.attn, block.norm1(x), mask_i))
        x = x + block.ls2(block.mlp(block.norm2(x)))
        return x
    
    def attn(self, block, x, mask_i=None):
        # forward_orig.__self__
        B, N, C = x.shape
        qkv = block.qkv(x).reshape(B, N, 3, block.num_heads, C // block.num_heads).permute(2, 0, 3, 1, 4)
        
        q, k, v = qkv[0] * block.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) # [B, heads, 1+HW, 1+HW]

        if mask_i is not None:
            attn[:, :, :, mask_i] = float("-inf")
        attn = attn.softmax(dim=-1)
        attn = block.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = block.proj(x)
        x = block.proj_drop(x)

        return x 
      
   