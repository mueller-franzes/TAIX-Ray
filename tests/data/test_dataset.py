from cxr.data.datasets import CXR_Dataset 
import torch 
from pathlib import Path 
from torchvision.utils import save_image

def tensor2image(tensor, batch=0):
    return (tensor if tensor.ndim<5 else torch.swapaxes(tensor[batch], 0, 1).reshape(-1, *tensor.shape[-2:])[:,None])


ds = CXR_Dataset(
    # random_flip=True,
    random_center=True,
    # random_rotate=True, 
    # random_inverse=True,
)

print(f"Dataset Length", len(ds))


item = ds[20]
uid = item["uid"]
img = item['source']
label = item['target']

print("UID", uid, "Image Shape", list(img.shape), "Label", label)

path_out = Path.cwd()/'results/test'
path_out.mkdir(parents=True, exist_ok=True)
img = tensor2image(img[None])
save_image(img, path_out/f'test.png', normalize=True)
