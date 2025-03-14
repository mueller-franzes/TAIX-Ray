
from pathlib import Path
from datetime import datetime
import argparse
import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from cxr.data.datasets import CXR_Dataset
from cxr.data.datamodules import DataModule
from cxr.models import ResNet, ResNetRegression, MST, MSTRegression


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, default=Path.cwd())
    parser.add_argument('--model', type=str, default='MST', choices=['ResNet', 'MST']) 
    parser.add_argument('--task', type=str, default="binary", choices=['multilabel', 'multiclass', 'binary', 'ordinal', 'absolute'])
    parser.add_argument('--label', type=str, default="none", choices=list(CXR_Dataset.CLASS_LABELS.keys())+['none'])
    parser.add_argument('--regression', action='store_true')
    args = parser.parse_args()
    regression = args.regression
    label = args.label if args.label != 'none' else None

    #------------ Settings/Defaults ----------------
    torch.set_float32_matmul_precision('high')
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path(args.run_dir) / 'runs'  / f'{args.model}_{current_time}'
    path_run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    

    # ------------ Load Data ----------------
    ds_train = CXR_Dataset(split='train', regression=regression, label=label, cache_images=True, random_center=True, random_ver_flip=True, random_rotate=True, random_inverse=False)
    ds_val = CXR_Dataset(split='val', regression=regression, label=label, cache_images=True)
    
    samples = len(ds_train) + len(ds_val)
    batch_size = 32
    accumulate_grad_batches = 1 
    steps_per_epoch = samples / batch_size / accumulate_grad_batches

    weights = None 
    if label is not None:
        class_counts = ds_train.df[label].value_counts()
        class_weights = 1 / class_counts / len(class_counts)
        weights = ds_train.df[label].map(lambda x: class_weights[x]).values
    


    dm = DataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_val,
        batch_size=batch_size, 
        pin_memory=True,
        weights=weights, 
        num_workers=16,
    )

    # ------------ Initialize Model ------------
    loss_kwargs = {}
    out_ch = len(ds_train.label)
    if regression and (args.task== "ordinal"):
        out_ch = sum(ds_train.class_labels_num)  
        loss_kwargs={'class_labels_num': ds_train.class_labels_num} 

    model_map = {
        'ResNet': ResNetRegression if regression else ResNet,
        'MST': MSTRegression if regression else MST
    }
    MODEL = model_map.get(args.model, None)
    model = MODEL(
        in_ch=1, 
        out_ch=out_ch,
        task= args.task, 
        loss_kwargs=loss_kwargs
    )


    # -------------- Training Initialization ---------------
    to_monitor = "val/MAE" if regression else "val/AUC_ROC"
    min_max = "min" if regression else "max"
    log_every_n_steps = 50
    logger = WandbLogger(project='CXR', name=f"{type(model).__name__}_{current_time}_{args.label}_large", log_model=False)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0, # minimum change in the monitored quantity to qualify as an improvement
        patience=10, # number of checks with no improvement
        mode=min_max
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir), # dirpath
        monitor=to_monitor,
        # every_n_train_steps=log_every_n_steps,
        save_last=True,
        save_top_k=1,
        mode=min_max,
    )
    trainer = Trainer(
        accelerator=accelerator,
        accumulate_grad_batches=accumulate_grad_batches,
        precision='16-mixed',
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing, early_stopping, lr_monitor],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=log_every_n_steps,
        max_epochs=1000,
        num_sanity_val_steps=2,
        logger=logger
    )
    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=dm)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(path_run_dir, checkpointing.best_model_path)


