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

    cmap = cmap_dict[label.split('_')[0]]  
    #  -------------------------- Confusion Matrix -------------------------
    y_scores_bin = y_score>=0.5 # WANRING: Must be >= not > 
    cm = confusion_matrix(y_true, y_scores_bin) # [[TN, FP], [FN, TP]]
    acc = accuracy_score(y_true, y_scores_bin)
    df_cm = pd.DataFrame(data=cm, columns=['False', 'True'], index=['False', 'True'])
    
    sns.heatmap(df_cm, ax=axis, cmap=cmap, cbar=False, fmt='d', annot=True) 
    axis.set_title(f'{label}', fontdict=fontdict) # CM =  [[TN, FP], [FN, TP]]  \nACC={acc:.2f}
    axis.set_xlabel('Neural Network' , fontdict=fontdict)
    axis.set_ylabel('Physicians' , fontdict=fontdict)

    return cm

path_run = Path('results/MST_2025_03_12_115000_w_aug')

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