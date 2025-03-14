import torch 
from tqdm import tqdm

from cxr.models import MST, MSTRegression
from cxr.models import ResNet, ResNetRegression
from cxr.data.datasets import CXR_Dataset
from cxr.data.datamodules import DataModule 

# label = None 
label = None #'HeartSize'
regression = True
task = "ordinal"
ds_train = CXR_Dataset(split='test', label=label, regression=regression)


device=torch.device(f'cuda:0')

loss_kwargs = {}
out_ch = len(ds_train.label)
if regression and (task== "ordinal"):
    out_ch = sum(ds_train.class_labels_num)  
    loss_kwargs={'class_labels_num': ds_train.class_labels_num} 


if label is not None:
    class_counts = ds_train.df[label].value_counts()
    class_weights = 1 / class_counts / len(class_counts)
    weights = ds_train.df[label].map(lambda x: class_weights[x]).values

MODEL = ResNetRegression if regression else MST
model = MODEL(
    in_ch=1, 
    out_ch=out_ch,
    task= task, 
    loss_kwargs=loss_kwargs
)


model.to(device)
model.eval()

dm = DataModule(ds_train=ds_train, batch_size=3, num_workers=0)
dl = dm.train_dataloader() 


for idx, batch in tqdm(enumerate(iter(dl))):
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    loss = model._step(batch, batch_idx=idx, state="train", step=idx*dm.batch_size)
    print("loss", loss)

