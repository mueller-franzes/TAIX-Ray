
from pathlib import Path
from datetime import datetime
import argparse
import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from cxr.data.datasets import CXR_Dataset
from cxr.data.datamodules import DataModule
from cxr.models import ResNet
from cxr.models import MST


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, default=Path.cwd())
    parser.add_argument('--model', type=str, default='ResNet', choices=['ResNet', 'MST']) 
    parser.add_argument('--task', type=str, default="multilabel", choices=['multilabel', 'multiclass'])
    args = parser.parse_args()

    #------------ Settings/Defaults ----------------
    torch.set_float32_matmul_precision('high')
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path(args.run_dir) / 'runs'  / f'{args.model}_{current_time}'
    path_run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'


    # ------------ Load Data ----------------
    ds_train = CXR_Dataset(split='train', cache_images=True, random_center=True, random_ver_flip=True, random_rotate=True)
    ds_val = CXR_Dataset(split='val', cache_images=True)
    
    samples = len(ds_train) + len(ds_val)
    batch_size = 32
    accumulate_grad_batches = 1 
    steps_per_epoch = samples / batch_size / accumulate_grad_batches

    if args.task == "multiclass":
        class_counts = ds_train.df[ds_train.label].value_counts()
        class_weights = 0.5 / class_counts
        weights = ds_train.df[ds_train.label].map(lambda x: class_weights[x]).values
    else:
        weights = None

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
    out_ch = 2 if args.task=="multiclass" else len(ds_train.label)
    MODEL = ResNet if args.model == 'ResNet' else MST
    model = MODEL(
        in_ch=1, 
        out_ch=out_ch,
        task=args.task
    )

    # Load pretrained model 
    # model = ResNet.load_from_checkpoint('runs/DUKE/2024_11_14_132823/epoch=41-step=17514.ckpt')


    # -------------- Training Initialization ---------------
    to_monitor = "val/AUC_ROC"  
    min_max = "max"
    log_every_n_steps = 50
    logger = WandbLogger(project='CXR', name=f"{type(model).__name__}_{current_time}", log_model=False)

    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0, # minimum change in the monitored quantity to qualify as an improvement
        patience=25, # number of checks with no improvement
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
        callbacks=[checkpointing, early_stopping],
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


