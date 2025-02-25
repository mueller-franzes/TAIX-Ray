import torch 
from tqdm import tqdm

from cxr.models import MST

from cxr.data.datasets import CXR_Dataset
from cxr.data.datamodules import DataModule 

label = None 
ds_test = CXR_Dataset(split='test', label=label)


device=torch.device(f'cuda:0')

task = task="multilabel" if label is None else "multiclass"
out_ch = 2 if task=="multiclass" else len(ds_test.label)
model = MST(in_ch=1, out_ch=out_ch, task=task)
model.to(device)
model.eval()

dm = DataModule(ds_test=ds_test, batch_size=2, num_workers=0)
dl = dm.test_dataloader() 


for idx, batch in tqdm(enumerate(iter(dl))):
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    loss = model._step(batch, batch_idx=idx, state="train", step=idx*dm.batch_size)
    print("loss", loss)

