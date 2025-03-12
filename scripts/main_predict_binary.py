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
from torchvision.utils import save_image

from cxr.data.datasets import CXR_Dataset
from cxr.data.datamodules import DataModule
from cxr.models import MST
from cxr.utils.roc_curve import plot_roc_curve, cm2acc, cm2x
from cxr.models.utils.functions import tensor2image, tensor_cam2image, minmax_norm


def evaluate(gt, nn, nn_pred, label, path_out):
    fontdict = {'fontsize': 10, 'fontweight': 'bold'}    
    # ------------------------------- ROC-AUC ---------------------------------
    fig, axis = plt.subplots(ncols=1, nrows=1, figsize=(6,6)) 
    y_pred_lab = np.asarray(nn_pred)
    y_true_lab = np.asarray(gt)
    tprs, fprs, auc_val, thrs, opt_idx, cm = plot_roc_curve(y_true_lab, y_pred_lab, axis, name=f"AUC {label}", fontdict=fontdict)
    fig.tight_layout()
    fig.savefig(path_out/f'roc_{label}.png', dpi=300)
    logger.info("AUC {:.2f}".format(auc_val))

    #  -------------------------- Confusion Matrix -------------------------
    acc = cm2acc(cm)
    _,_, sens, spec = cm2x(cm)
    df_cm = pd.DataFrame(data=cm, columns=['False', 'True'], index=['False', 'True'])
    fig, axis = plt.subplots(1, 1, figsize=(4,4))
    sns.heatmap(df_cm, ax=axis, cbar=False, fmt='d', annot=True) 
    axis.set_title(f'Confusion Matrix {label} ACC={acc:.2f}', fontdict=fontdict) # CM =  [[TN, FP], [FN, TP]] 
    axis.set_xlabel('Prediction' , fontdict=fontdict)
    axis.set_ylabel('True' , fontdict=fontdict)
    fig.tight_layout()
    fig.savefig(path_out/f'confusion_matrix_{label}.png', dpi=300)

    logger.info(f"------Label {label}--------")
    logger.info(f"Number of GT=1: {np.sum(y_true_lab)}")
    logger.info("Confusion Matrix {}".format(cm))
    logger.info("Sensitivity {:.2f}".format(sens))
    logger.info("Specificity {:.2f}".format(spec))



if __name__ == "__main__":
    #------------ Get Arguments ----------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_run', default='runs/MST_2025_03_02_165634_w_aug/epoch=19-step=86240.ckpt', type=str)
    parser.add_argument('--label', default='none', type=lambda x: None if x.lower() == 'none' else x) # None will use all labels 
    parser.add_argument('--show_attention', action='store_true')
    args = parser.parse_args()
    show_attention = args.show_attention
    batch_size = 32

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
    ds_test = CXR_Dataset(label=args.label, split='test')


    dm = DataModule(
        ds_test = ds_test,
        batch_size=batch_size, 
        num_workers=16,
        # pin_memory=True,
    ) 


    # ------------ Initialize Model ------------
    model = MST.load_from_checkpoint(path_run)
    model.to(device)
    model.eval()


    # ------------ Predict ----------------
    results = []
    for batch in tqdm(dm.test_dataloader()):
        uid, source, target = batch['uid'], batch['source'], batch['target']

        if not show_attention:
            # Run Model 
            with torch.no_grad():
                pred = model(source.to(device)).cpu()

        else:
            pred, weight = model.forward_attention(source.to(device))
            pred = pred.cpu()
            weight = weight.cpu().detach()

            path_out2 = path_out/'weights'
            path_out2.mkdir(parents=True, exist_ok=True)
            save_image(tensor2image(source), path_out2/f'input_{uid[0]}.png', normalize=True)
            
            # Iterate over labels 
            for i in range(weight.size(-1)):
                weight_i = weight[:, :, i] # [B, HW]
                weight_i = weight_i.reshape(batch_size, 32, 32) # [B, H, W]
                weight_i = weight_i[:, None] # [B, C, H, W]

                weight_i = F.interpolate(weight_i, size=source.shape[2:], mode='bilinear') # trilinear, area
                # weight_i = weight_i/weight_i.sum(-1).sum(-1).unsqueeze(-1).unsqueeze(-1)
                save_image(tensor_cam2image(minmax_norm(source), minmax_norm(weight_i), alpha=0.5),  path_out2/f'overlay_{uid[0]}_{ds_test.label[i]}.png', normalize=False)
       



        pred_prob = torch.sigmoid(pred) # [B, Classes]
        pred = (pred_prob>0.5).type(torch.int)
        
        for b in range(pred.size(0)):
            results.append({
                'UID': uid[b],
                'GT': target[b].tolist(),
                'NN_prob': pred_prob[b].tolist(),
                'NN': pred[b].tolist(),
            })

    # ------------ Save Results ----------------
    df = pd.DataFrame(results)
    df.to_csv(path_out/'results.csv', index=False)


    # ------------ Evaluate ----------------
    # df = pd.read_csv(path_out/'results.csv')
    # df['GT'] = df['GT'].apply(ast.literal_eval)
    # df['NN'] = df['NN'].apply(ast.literal_eval)
    # df['NN_prob'] = df['NN_prob'].apply(ast.literal_eval)

    gt = np.stack(df['GT'].values)
    nn = np.stack(df['NN'].values)
    nn_pred = np.stack(df['NN_prob'].values) 
    for i in range(gt.shape[1]):
        evaluate(gt[:, i], nn[:, i], nn_pred[:, i], ds_test.LABELS[i], path_out)



