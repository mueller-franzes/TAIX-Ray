from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

from cxr.data.datasets import CXR_Dataset
from cxr.utils.roc_curve import bootstrap_metric

def cohen_kappa_score_linear(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="linear")

def evaluate(y_true, y_pred, label, label_vals, axis):
    fontdict = {'fontsize': 10, 'fontweight': 'bold'}    
    cmap = cmap_dict[label.split('_')[0]]  

    #  -------------------------- Agreement -------------------------
    kappa = cohen_kappa_score_linear(y_true, y_pred)
    kappa_mean, kappa_ci, kappa_std = bootstrap_metric(y_true, y_pred, cohen_kappa_score_linear)
    print(f"{label}: Cohen's Kappa = {kappa:.2f} [{kappa_ci[0]:.2f} - {kappa_ci[1]:.2f}], STD = {kappa_std:.2f}")

    #  -------------------------- Confusion Matrix -------------------------
    cm = confusion_matrix(y_true, y_pred) # [[TN, FP], [FN, TP]]
    acc = accuracy_score(y_true, y_pred)
    df_cm = pd.DataFrame(data=cm, columns=label_vals, index=label_vals)
    
    sns.heatmap(df_cm, ax=axis, cmap=cmap, cbar=False, fmt='d', annot=True) 
    axis.set_title(f'{label}', fontdict=fontdict) # CM =  [[TN, FP], [FN, TP]]   ACC={acc:.2f}
    axis.set_xlabel('Neural Network' , fontdict=fontdict)
    axis.set_ylabel('Radiologist' , fontdict=fontdict)

    if label == "HeartSize":
        plt.setp(axis.get_xticklabels(), rotation=25, ha='center')

    acc_mean, acc_ci, acc_std = bootstrap_metric(y_true, y_pred, accuracy_score)
    print(f"{label}: Accuracy = {acc:.2f} [{acc_ci[0]:.2f} - {acc_ci[1]:.2f}], STD = {acc_std:.2f}")


    return cm


path_run = Path('results/MST_2025_03_20_135914_ordinal')

df = pd.read_csv(path_run/'results.csv')
df['GT'] = df['GT'].apply(ast.literal_eval)
df['NN'] = df['NN'].apply(ast.literal_eval)

gt = np.stack(df['GT'].values)
nn = np.stack(df['NN'].values)

cmap_dict = {
    'HeartSize': plt.cm.Reds, 
    'PulmonaryCongestion': plt.cm.Blues, 
    'PleuralEffusion': plt.cm.Greens, 
    'PulmonaryOpacities': plt.cm.Oranges, 
    'Atelectasis': plt.cm.Purples,
}

CLASS_LABELS = CXR_Dataset.CLASS_LABELS
labels = list(CLASS_LABELS.keys())

nrows = 2
ncols = 4
fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*3,nrows*3))
ax = ax.flatten()
for i in range(gt.shape[1]):
    label = labels[i]
    label_vals = CLASS_LABELS[label]
    evaluate(gt[:, i], nn[:, i], label, label_vals, ax[i])
fig.tight_layout()
plt.subplots_adjust( hspace=0.7) # wspace=0.7,
fig.savefig(path_run/'confusion_matrices_All.png', dpi=300)