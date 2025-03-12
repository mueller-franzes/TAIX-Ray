
import torch 
import torch.nn as nn 
import torchvision.models as models
import  torch.optim.lr_scheduler as lr_scheduler
from transformers import AutoModel

from .base_model import BasicClassifier, BasicRegression

def attn(block, x, mask_i=None):
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

def run_block(block, x, mask_i):
    x = x + block.ls1(attn(block.attn, block.norm1(x), mask_i))
    x = x + block.ls2(block.mlp(block.norm2(x)))
    return x
    


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

    def forward(self, x_in, **kwargs):
        x = x_in.to(self.device) # [B, 1, H, W]
        B, *_ = x.shape
        x = x.repeat(1, 3, 1, 1) # Gray to RGB
        x = self.model(x) #  -> [B, out] 
        x = self.linear(x)
        return x
    




class MST(BasicClassifier):
    def __init__(
        self, 
        in_ch, 
        out_ch, 
        task="multilabel",
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
        # self.model = AutoModel.from_pretrained("microsoft/rad-dino")

        # emb_ch = 768 
        emb_ch = self.model.num_features 
        self.linear = nn.Linear(emb_ch, out_ch)



    def forward(self, x_in, save_attn=False, **kwargs):
        x = x_in.to(self.device) # [B, 1, H, W]
        B, *_ = x.shape

        if save_attn:
            # fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
            # torch.backends.mha.set_fastpath_enabled(False)
            self.attention_maps_slice = []
            self.attention_maps = []
            self.hooks = []
            self.register_hooks()

        x = x.repeat(1, 3, 1, 1) # Gray to RGB
        
    
        if save_attn:
            x = self.model(x, is_training=True)
            # torch.backends.mha.set_fastpath_enabled(fastpath_enabled)
            self.deregister_hooks()
            return x 

        x = self.model(x) #  -> [B, out] 
        # x = x.pooler_output
        x = self.linear(x)
        return x
    


        


    def forward_mask(self, block, x, mask_i):
        """
        Recompute the class token representation with the contribution of one patch token removed.
        """
        x = run_block(block, x, mask_i)
        x_norm = self.model.norm(x)
        x_norm_clstoken = x_norm[:, 0]
        pred = self.linear(x_norm_clstoken)
        return pred
    
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

        # with torch.no_grad():
        #     x = self.forward(x_in, save_attn=True)
        #     hidden_state = x['x_norm_patchtokens'] # [B, tokens, E]
        #     x = x['x_norm_clstoken']
        #     pred = self.linear(x) # [B, out]


        # contributions = self.linear(hidden_state).sigmoid() # (B, HW, num_labels)
        # cls_attention = self.get_plane_attention() # [B, HW]   

        # # Attention of the labels to the CLS token
        # token_relevance = cls_attention.unsqueeze(-1) * contributions
        # token_relevance /= token_relevance.sum(1, keepdim=True)

        # return pred, token_relevance

        return pred, token_relevance
    
    
      
   
    def get_plane_attention(self):
        attention_map_dino = self.attention_maps[-1] # [B, Heads, 1+HW, 1+HW]
        attention_map_dino = attention_map_dino.mean(dim=1)  # -> [B, 1+HW, 1+HW]
        num_register_tokens =  0
        img_slice = slice(num_register_tokens+1, None) 
        attention_map_dino = attention_map_dino[:, 0, img_slice] # -> [B, HW]
        attention_map_dino /= attention_map_dino.sum(dim=-1, keepdim=True) # Normalize 
        return attention_map_dino #  [B, HW]
    

    
    def register_hooks(self):
        # ------------------------- DINOv2 attention -----------------
        def enable_attention_dino(mod):
                forward_orig = mod.forward
                def forward_wrap(self2, x):
                    # forward_orig.__self__
                    B, N, C = x.shape
                    qkv = self2.qkv(x).reshape(B, N, 3, self2.num_heads, C // self2.num_heads).permute(2, 0, 3, 1, 4)
                    
                    q, k, v = qkv[0] * self2.scale, qkv[1], qkv[2]
                    attn = q @ k.transpose(-2, -1)
           
                    attn = attn.softmax(dim=-1)
                    attn = self2.attn_drop(attn)

                    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x = self2.proj(x)
                    x = self2.proj_drop(x)

                    # Hook attention map 
                    self.attention_maps.append(attn)

                    return x
                
                mod.forward = lambda x: forward_wrap(mod, x)
                mod.foward_orig = forward_orig

        # Hook Dino Attention
        for name, mod in self.model.named_modules():
            if name.endswith('.attn'):
                enable_attention_dino(mod)



    def deregister_hooks(self):
        for handle in self.hooks:
            handle.remove()

        # ------------------------- DINOv2 attention -----------------
        for name, mod in self.model.named_modules():
            if name.endswith('.attn'):
                mod.forward = mod.foward_orig
