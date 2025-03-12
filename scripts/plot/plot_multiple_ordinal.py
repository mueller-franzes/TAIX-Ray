from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score

from cxr.data.datasets import CXR_Dataset



def evaluate(y_true, y_pred, label, label_vals, axis):
    fontdict = {'fontsize': 10, 'fontweight': 'bold'}    

    kappa = cohen_kappa_score(y_true, y_pred, weights="linear")

    print(f"{kappa:.2f}")

    #  -------------------------- Confusion Matrix -------------------------
    cm = confusion_matrix(y_true, y_pred) # [[TN, FP], [FN, TP]]
    acc = accuracy_score(y_true, y_pred)
    df_cm = pd.DataFrame(data=cm, columns=label_vals, index=label_vals)
    
    sns.heatmap(df_cm, ax=axis, cmap="Blues", cbar=False, fmt='d', annot=True) 
    axis.set_title(f'{label} ACC={acc:.2f}', fontdict=fontdict) # CM =  [[TN, FP], [FN, TP]] 
    axis.set_xlabel('Prediction' , fontdict=fontdict)
    axis.set_ylabel('True' , fontdict=fontdict)

    return cm


path_run = Path('results/MST_2025_03_05_151455_reg_corn')

df = pd.read_csv(path_run/'results.csv')
df['GT'] = df['GT'].apply(ast.literal_eval)
df['NN'] = df['NN'].apply(ast.literal_eval)

gt = np.stack(df['GT'].values)
nn = np.stack(df['NN'].values)

CLASS_LABELS = CXR_Dataset.CLASS_LABELS
labels = list(CLASS_LABELS.keys())

nrows = 2
ncols = 4
fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*4,nrows*4))
ax = ax.flatten()
for i in range(gt.shape[1]):
    label = labels[i]
    label_vals = CLASS_LABELS[label]
    evaluate(gt[:, i], nn[:, i], label, label_vals, ax[i])
fig.tight_layout()
fig.savefig('confusion_matrices.png', dpi=300)