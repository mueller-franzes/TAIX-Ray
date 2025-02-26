
import torch 
import torch.nn as nn 
import torchvision.models as models
import  torch.optim.lr_scheduler as lr_scheduler


from .base_model import BasicClassifier




class MST(BasicClassifier):
    def __init__(
        self, 
        in_ch, 
        out_ch, 
        task="multilabel",
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
        x = self.model(x) #  -> [B, out] 
        # x = self.model(x, is_training=True)


        if save_attn:
            # torch.backends.mha.set_fastpath_enabled(fastpath_enabled)
            self.deregister_hooks()
            return x 
        
        x = self.linear(x)
        return x
    
    def forward_attention(self, x_in, target_class=0):
        with torch.no_grad():
            x = self.forward(x_in, save_attn=True)
            hidden_state = x['x_norm_patchtokens'] 
            x = x['x_norm_clstoken']
            pred = self.linear(x) # [B, out]

        contributions = hidden_state[0]@self.linear.weight.T # (seq_len, num_labels)
        cls_attention = self.get_plane_attention() # [B, HW]   # (batch, seq_len)

        # Attention of the labels to the CLS token
        token_relevance = cls_attention.T * contributions

        token_relevance -= token_relevance.min(dim=0).values
        token_relevance /= token_relevance.sum(dim=0)

        return pred, token_relevance

        # # Create gradients 
        # x.requires_grad_()  # Enable gradient tracking for x
        # self.linear.zero_grad()  # Zero out previous grads
        # pred = self.linear(x)   # Get the prediction
        # pred_target = pred[:, target_class] # Select the target class  # Create a tensor of shape [B, out]

        # # Backpropagate for the target class
        # pred_target.backward()

        # # Get gradients of x (feature map before linear)
        # attention_label2cls = x.grad  # Shape: [B, Labels]
        # attention_label2cls = attention_label2cls.unsqueeze(-1) # [B, Labels, 1]

        # # Attention of the CLS token to the input patches
        # attention_cls2patch = self.get_plane_attention() # [B, HW]
        # attention_cls2patch = attention_cls2patch.unsqueeze(1) # [B, 1, HW]

        # # Attention of the labels to the CLS token
        # attention = attention_label2cls*attention_cls2patch # [B, Labels, HW]
        
        # return pred, attention  
   
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
