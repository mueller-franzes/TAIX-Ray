
import torch 
from cxr.data.datasets import CXR_Dataset 
from cxr.models.utils.losses import CornLossMulti, CELossMulti



ds = CXR_Dataset(
    regression=True,
    label=None,
)

# class_labels_num = [len(ds.CLASS_LABELS[l])-1 for l in ds.label]
class_labels_num = [len(ds.CLASS_LABELS[l]) for l in ds.label]

# loss = CornLossMulti(class_labels_num=class_labels_num)
loss = CELossMulti(class_labels_num=class_labels_num)

item = ds[0]
target = torch.from_numpy(item['target'][None])

pred = torch.rand((1, sum(class_labels_num) )) # [B, C*num]

m = loss(pred, target)

pred_lables = loss.logits2labels(pred)