from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score

from cxr.data.datasets import CXR_Dataset
from cxr.utils.roc_curve import bootstrap_metric, auc_bootstrapping, compute_stats


def evaluate(y_true, y_score, label, axis):
    fontdict = {'fontsize': 10, 'fontweight': 'bold'}  

    # ------------------------------- ROC-AUC ---------------------------------
    fprs, tprs, thrs = roc_curve(y_true, y_score, drop_intermediate=False)
    auc_val = auc(fprs, tprs)
    tprs, aucs, thrs, mean_fpr = auc_bootstrapping(y_true, y_score)
    auc_mean, auc_ci, auc_std = compute_stats(aucs)
    print(f"{label}: AUC = {auc_val:.2f} [{auc_ci[0]:.2f} - {auc_ci[1]:.2f}], STD = {auc_std:.2f}")

    cmap = cmap_dict[label.split('_')[0]]  
    #  -------------------------- Confusion Matrix -------------------------
    y_scores_bin = y_score>=0.5 # WANRING: Must be >= not > 
    cm = confusion_matrix(y_true, y_scores_bin) # [[TN, FP], [FN, TP]]
    acc = accuracy_score(y_true, y_scores_bin)
    df_cm = pd.DataFrame(data=cm, columns=['Absent', 'Present'], index=['Absent', 'Present'])
    
    sns.heatmap(df_cm, ax=axis, cmap=cmap, cbar=False, fmt='d', annot=True) 
    label_str = "Cardiomegaly" if label == 'HeartSize' else label
    axis.set_title(f'{label_str}', fontdict=fontdict) # CM =  [[TN, FP], [FN, TP]]  \nACC={acc:.2f}
    axis.set_xlabel('Neural Network' , fontdict=fontdict)
    axis.set_ylabel('Radiologist' , fontdict=fontdict)

    acc_mean, acc_ci, acc_std = bootstrap_metric(y_true, y_scores_bin, accuracy_score)
    print(f"{label}: Accuracy = {acc:.2f} [{acc_ci[0]:.2f} - {acc_ci[1]:.2f}], STD = {acc_std:.2f}")

    return cm

path_run = Path('results/MST_2025_03_18_101740_binary')

cmap_dict = {
    'HeartSize': plt.cm.Reds, 
    'PulmonaryCongestion': plt.cm.Blues, 
    'PleuralEffusion': plt.cm.Greens, 
    'PulmonaryOpacities': plt.cm.Oranges, 
    'Atelectasis': plt.cm.Purples,
}

df = pd.read_csv(path_run/'results.csv')
df['GT'] = df['GT'].apply(ast.literal_eval)
df['NN'] = df['NN'].apply(ast.literal_eval)
df['NN_prob'] = df['NN_prob'].apply(ast.literal_eval)

gt = np.stack(df['GT'].values)
nn = np.stack(df['NN'].values)
nn_pred = np.stack(df['NN_prob'].values) 

labels = list(CXR_Dataset.CLASS_LABELS.keys())

nrows = 2
ncols = 4
fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*2.5,nrows*2.5))
ax = ax.flatten()
for i in range(gt.shape[1]):
    evaluate(gt[:, i], nn_pred[:, i], labels[i], ax[i])

fig.tight_layout()
plt.subplots_adjust(wspace=0.7, hspace=0.7)


fig.savefig(path_run/'confusion_matrix_All.png', dpi=300)