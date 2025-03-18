import argparse
from pathlib import Path
import logging
from tqdm import tqdm
import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
import ast 
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

from cxr.data.datasets import CXR_Dataset
from cxr.data.datamodules import DataModule
from cxr.models import MSTRegression



def evaluate(gt, nn, label, label_vals, path_out):
    fontdict = {'fontsize': 10, 'fontweight': 'bold'}    
    y_pred_lab = np.asarray(nn)
    y_true_lab = np.asarray(gt)

    #  -------------------------- Confusion Matrix -------------------------
    cm = confusion_matrix(y_true_lab, y_pred_lab, labels=list(range(len(label_vals))))
    acc = accuracy_score(y_true_lab, y_pred_lab)
    df_cm = pd.DataFrame(data=cm, columns=label_vals, index=label_vals)
    fig, axis = plt.subplots(1, 1, figsize=(4,4))
    sns.heatmap(df_cm, ax=axis, cbar=False, cmap="Blues", fmt='d', annot=True) 
    axis.set_title(f'{label} ACC={acc:.2f}', fontdict=fontdict) # CM =  [[TN, FP], [FN, TP]] 
    axis.set_xlabel('Prediction' , fontdict=fontdict)
    axis.set_ylabel('True' , fontdict=fontdict)
    fig.tight_layout()
    fig.savefig(path_out/f'confusion_matrix_{label}.png', dpi=300)

    #  -------------------------- Agreement -------------------------
    kappa = cohen_kappa_score(y_true_lab, y_pred_lab, weights="linear") 
    print(label, "Kappa", kappa)


if __name__ == "__main__":
    #------------ Get Arguments ----------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', default='runs/MST_2025_03_13_220252/epoch=10-step=47289.ckpt', type=str)
    parser.add_argument('--label', default='none', type=lambda x: None if x.lower() == 'none' else x) # None will use all labels 
    args = parser.parse_args()
    batch_size = 16

    #------------ Settings/Defaults ----------------
    path_run = Path(args.path_run) 
    run_name = path_run.parent.name
    path_out = Path().cwd()/'results'/run_name
    path_out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ------------ Logging --------------------
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


    # ------------ Load Data ----------------
    ds_test = CXR_Dataset(label=args.label, regression=True, split='test')


    dm = DataModule(
        ds_test = ds_test,
        batch_size=batch_size, 
        num_workers=16,
        # pin_memory=True,
    ) 


    # ------------ Initialize Model ------------
    model = MSTRegression.load_from_checkpoint(path_run)
    model.to(device)
    model.eval()


    # ------------ Predict ----------------
    results = []
    for batch in tqdm(dm.test_dataloader()):
        uid, source, target = batch['uid'], batch['source'], batch['target']

        with torch.no_grad():
            pred = model(source.to(device)).cpu()

        # Transfer logits to integer 
        if model.task == 'absolute':
            pred = pred.round().to(int)
        elif model.task == 'ordinal':
            pred = model.loss_func.logits2labels(pred)
        
        for b in range(pred.size(0)):
            results.append({
                'UID': uid[b],
                'GT': target[b].tolist(),
                'NN': pred[b].tolist(),
            })

    # ------------ Save Results ----------------
    df = pd.DataFrame(results)
    df.to_csv(path_out/'results.csv', index=False)


    # ------------ Evaluate ----------------
    # df = pd.read_csv(path_out/'results.csv')
    # df['GT'] = df['GT'].apply(ast.literal_eval)
    # df['NN'] = df['NN'].apply(ast.literal_eval)

    gt = np.stack(df['GT'].values)
    nn = np.stack(df['NN'].values)
    for i in range(gt.shape[1]):
        label = ds_test.label[i]
        label_vals = ds_test.CLASS_LABELS[label]
        evaluate(gt[:, i], nn[:, i], label, label_vals, path_out)



