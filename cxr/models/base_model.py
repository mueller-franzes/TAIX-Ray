from pathlib import Path
import json
import torch 
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import AUROC, Accuracy, MeanAbsoluteError
from .utils.losses import CornLossMulti

class VeryBasicModel(pl.LightningModule):
    def __init__(self, save_hyperparameters=True):
        super().__init__()
        if save_hyperparameters:
            self.save_hyperparameters()
        self._step_train = -1
        self._step_val = -1
        self._step_test = -1


    def forward(self, x, cond=None):
        raise NotImplementedError


    def _step(self, batch: dict, batch_idx: int, state: str, step: int):
        raise NotImplementedError
    
    def _epoch_end(self, state:str):
        return 

    def training_step(self, batch: dict, batch_idx: int ):
        self._step_train += 1 
        return self._step(batch, batch_idx, "train", self._step_train)

    def validation_step(self, batch: dict, batch_idx: int):
        self._step_val += 1
        return self._step(batch, batch_idx, "val", self._step_val )

    def test_step(self, batch: dict, batch_idx: int):
        self._step_test += 1
        return self._step(batch, batch_idx, "test", self._step_test)

    def on_train_epoch_end(self) -> None: 
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")



    @classmethod
    def save_best_checkpoint(cls, path_checkpoint_dir, best_model_path):
        with open(Path(path_checkpoint_dir) / 'best_checkpoint.json', 'w') as f:
            json.dump({'best_model_epoch': Path(best_model_path).name}, f)

    @classmethod
    def _get_best_checkpoint_path(cls, path_checkpoint_dir, **kwargs):
        with open(Path(path_checkpoint_dir) / 'best_checkpoint.json', 'r') as f:
            path_rel_best_checkpoint = Path(json.load(f)['best_model_epoch'])
        return Path(path_checkpoint_dir)/path_rel_best_checkpoint

    @classmethod
    def load_best_checkpoint(cls, path_checkpoint_dir, **kwargs):
        path_best_checkpoint = cls._get_best_checkpoint_path(path_checkpoint_dir)
        return cls.load_from_checkpoint(path_best_checkpoint, **kwargs)

    def load_pretrained(self, checkpoint_path, map_location=None, **kwargs):
        if checkpoint_path.is_dir():
            checkpoint_path = self._get_best_checkpoint_path(checkpoint_path, **kwargs)  

        checkpoint = torch.load(checkpoint_path, map_location=map_location)
     
        return self.load_weights(checkpoint["state_dict"], **kwargs)
    
    def load_weights(self, pretrained_weights, strict=True, **kwargs):
        filter = kwargs.get('filter', lambda key:key in pretrained_weights)
        init_weights = self.state_dict()
        pretrained_weights = {key: value for key, value in pretrained_weights.items() if filter(key)}
        init_weights.update(pretrained_weights)
        self.load_state_dict(init_weights, strict=strict)
        return self 




class BasicModel(VeryBasicModel):
    def __init__(
        self, 
        optimizer=torch.optim.Adam, 
        optimizer_kwargs={'lr':1e-3, 'weight_decay':1e-2},
        lr_scheduler= None, 
        lr_scheduler_kwargs={},
        save_hyperparameters=True
    ):
        super().__init__(save_hyperparameters=save_hyperparameters)
        if save_hyperparameters:
            self.save_hyperparameters()
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.lr_scheduler = lr_scheduler 
        self.lr_scheduler_kwargs = lr_scheduler_kwargs

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler is not None:
            lr_scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs)
            lr_scheduler_config  = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}
            return [optimizer], [lr_scheduler_config ]
        else:
            return [optimizer]



class BasicClassifier(BasicModel):
    def __init__(
        self, 
        in_ch,
        out_ch,
        task, # multiclass or multilabel or binary
        spatial_dims,
        loss = None,
        loss_kwargs = {},
        optimizer=torch.optim.AdamW, 
        optimizer_kwargs={'lr':1e-4, 'weight_decay':1e-2},
        lr_scheduler= None, 
        lr_scheduler_kwargs={},
        aucroc_kwargs={},
        acc_kwargs={},
        save_hyperparameters=True,
    ):
        super().__init__(optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs, save_hyperparameters)
        self.in_ch = in_ch 
        self.out_ch = out_ch 
        self.task = task 
        self.spatial_dims = spatial_dims

        if loss is not None:
            loss = loss 
        elif task == "multiclass":
            loss = torch.nn.CrossEntropyLoss
        elif task == "multilabel":
            loss = torch.nn.BCEWithLogitsLoss
        elif task == "binary":
            loss = torch.nn.BCEWithLogitsLoss
        else:
            raise ValueError("Unknown task and loss not provided")

        self.loss_func = loss(**loss_kwargs)
        self.loss_kwargs = loss_kwargs 

        if task == "multiclass":
            aucroc_kwargs.update({"task":"multiclass", 'num_classes':out_ch}) 
            acc_kwargs.update({"task":"multiclass", 'num_classes':out_ch}) 
        elif task == "multilabel":
            aucroc_kwargs.update({"task":"multilabel", 'num_labels':out_ch}) 
            acc_kwargs.update({"task":"multilabel", 'num_labels':out_ch}) 
        elif task == "binary":
            aucroc_kwargs.update({"task":"binary"}) 
            acc_kwargs.update({"task":"binary"}) 



        self.auc_roc = nn.ModuleDict({state:AUROC(**aucroc_kwargs) for state in ["train_", "val_", "test_"]}) # 'train' not allowed as key
        self.acc = nn.ModuleDict({state:Accuracy(**acc_kwargs) for state in ["train_", "val_", "test_"]})

    
    def _step(self, batch: dict, batch_idx: int, state: str, step: int):
        source = batch['source']
        target = batch['target'] # multiclass: [B,] multilabel [B, C]

        batch_size = source.shape[0]
        self.batch_size = batch_size 

        # Run Model 
        pred = self(source)

        # ------------------------- Compute Loss ---------------------------
        logging_dict = {}
        logging_dict['loss'] = self.compute_loss(pred, target)

        # --------------------- Compute Metrics  -------------------------------
        with torch.no_grad():
            # Aggregate here to compute for entire set later 
            self.acc[state+"_"].update(pred, target)
            self.auc_roc[state+"_"].update(pred, target) 
            
            # ----------------- Log Scalars ----------------------
            for metric_name, metric_val in logging_dict.items():
                self.log(f"{state}/{metric_name}", metric_val, batch_size=batch_size, on_step=True, on_epoch=True, 
                         sync_dist=False) 

        return logging_dict['loss'] 

    def _epoch_end(self, state):
        for name, value in [("ACC", self.acc[state+"_"]), ("AUC_ROC", self.auc_roc[state+"_"])]:
            self.log(f"{state}/{name}", value.compute(), batch_size=self.batch_size, on_step=False, on_epoch=True, 
                     sync_dist=True)
            value.reset()

    def compute_loss(self, pred, target):
        if self.task in ["multilabel", 'binary']:
            target = target.float()
        return self.loss_func(pred, target)
    


class BasicRegression(BasicModel):
    def __init__(
        self, 
        in_ch,
        out_ch,
        task, # ordinal, absolute
        spatial_dims,
        loss = None,
        loss_kwargs = {},
        optimizer=torch.optim.AdamW, 
        optimizer_kwargs={'lr':1e-4, 'weight_decay':1e-2},
        lr_scheduler= None, 
        lr_scheduler_kwargs={},
        save_hyperparameters=True,
    ):
        super().__init__(optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs, save_hyperparameters)
        self.in_ch = in_ch 
        self.out_ch = out_ch 
        self.task = task 
        self.spatial_dims = spatial_dims

        if loss is not None:
            loss = loss 
        elif task == "ordinal":
            loss = CornLossMulti
        elif task == "absolute":
            loss = nn.L1Loss
        else:
            raise ValueError("Unknown task")

        self.loss_func = loss(**loss_kwargs)
        self.loss_kwargs = loss_kwargs 


        self.mae = nn.ModuleDict({state:MeanAbsoluteError() for state in ["train_", "val_", "test_"]}) # 'train' not allowed as key

    
    def _step(self, batch: dict, batch_idx: int, state: str, step: int):
        source = batch['source']
        target = batch['target'] # ordinal: [B, num_classes]

        batch_size = source.shape[0]
        self.batch_size = batch_size 

        # Run Model 
        pred = self(source) # MAE expects [B, num_classes], CORN expects [B, num_classes*(num_labels-1)]

        # ------------------------- Compute Loss ---------------------------
        logging_dict = {}
        logging_dict['loss'] = self.compute_loss(pred, target)

        # --------------------- Compute Metrics  -------------------------------
        if self.loss_func.__class__.__name__ == "CornLossMulti":
            pred = self.loss_func.logits2labels(pred)

        with torch.no_grad():
            # Aggregate here to compute for entire set later 
            self.mae[state+"_"].update(pred, target)
            
            # ----------------- Log Scalars ----------------------
            for metric_name, metric_val in logging_dict.items():
                self.log(f"{state}/{metric_name}", metric_val, batch_size=batch_size, on_step=True, on_epoch=True, 
                         sync_dist=False) 

        return logging_dict['loss'] 

    def _epoch_end(self, state):
        for name, value in [("MAE", self.mae[state+"_"]), ]:
            self.log(f"{state}/{name}", value.compute(), batch_size=self.batch_size, on_step=False, on_epoch=True, 
                     sync_dist=True)
            value.reset()

    def compute_loss(self, pred, target):
        return self.loss_func(pred, target)