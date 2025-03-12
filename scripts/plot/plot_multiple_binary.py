from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast
from sklearn.metrics import confusion_matrix, accuracy_score

from cxr.data.datasets import CXR_Dataset



def evaluate(y_true, y_score, label, axis):
    fontdict = {'fontsize': 10, 'fontweight': 'bold'}    
    #  -------------------------- Confusion Matrix -------------------------
    y_scores_bin = y_score>=0.5 # WANRING: Must be >= not > 
    cm = confusion_matrix(y_true, y_scores_bin) # [[TN, FP], [FN, TP]]
    acc = accuracy_score(y_true, y_scores_bin)
    df_cm = pd.DataFrame(data=cm, columns=['False', 'True'], index=['False', 'True'])
    
    sns.heatmap(df_cm, ax=axis, cmap="Blues", cbar=False, fmt='d', annot=True) 
    axis.set_title(f'{label} ACC={acc:.2f}', fontdict=fontdict) # CM =  [[TN, FP], [FN, TP]] 
    axis.set_xlabel('Prediction' , fontdict=fontdict)
    axis.set_ylabel('True' , fontdict=fontdict)

    return cm

path_run = Path('results/MST_2025_03_02_165634_w_aug')

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
fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*3,nrows*3))
ax = ax.flatten()
for i in range(gt.shape[1]):
    evaluate(gt[:, i], nn_pred[:, i], labels[i], ax[i])
fig.tight_layout()
fig.savefig('aucs.png', dpi=300)