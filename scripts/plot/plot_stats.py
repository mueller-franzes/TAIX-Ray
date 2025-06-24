from pathlib import Path 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from cxr.data.datasets.cxr_dataset import CXR_Dataset
from matplotlib.ticker import FuncFormatter
CLASS_LABELS = CXR_Dataset.CLASS_LABELS


def format_with_dots(x, pos):
    return f'{x:,.0f}' 

def get_labels(cls):
    if cls == "HeartSize":
        return ['Normal', 'Borderline', 'Enlarged', 'Massively']
    # return ['None', 'Questionable', 'Mild', 'Moderate', 'Severe']
    return ['None', '(+)', '+', '++', '+++']



path_root = Path('/ocean_storage/data/UKA/UKA_Thorax')


path_out = Path.cwd()/'results'
path_out.mkdir(exist_ok=True)

path_root_pub = path_root/'public_export'
path_root_pub_meta = path_root_pub/'metadata' 

# Read CSV file
df_lab = pd.read_csv(path_root_pub_meta / 'annotation.csv')
df_lab['StudyDate'] = pd.to_datetime(df_lab['StudyDate'], format='%Y-%m-%d')
label_cols = list(CLASS_LABELS.keys()) # df_lab.columns[6:]

# ------------ General Statistics ------------
print(f"Total samples: {len(df_lab)}")

print("Mean Age {:.0f} ± {:.0f}".format(df_lab['Age'].mean()/365, df_lab['Age'].std()/365))
print("Range Age {:.0f} to {:.0f}".format(df_lab['Age'].min()/365, df_lab['Age'].max()/365))


patients = df_lab.groupby('PatientID')
num_patients = len(patients)
sex_counts = patients['Sex'].apply(lambda x:x.iloc[0]).value_counts()
print("Patients: ", num_patients)
print("Male", sex_counts['M'], f"({sex_counts['M']/num_patients*100:.0f}%)")
print("Female", sex_counts['F'], f"({sex_counts['F']/num_patients*100:.0f}%)")

physicians = df_lab.groupby('PhysicianID')
num_exam_per_phy = physicians['PhysicianID'].apply(len)
print(f"Number Physicians {len(physicians)}")
print(f"Avg. Exams per Physician {num_exam_per_phy.mean():.0f}")
print(f"Min-Max Exams per Physician: {num_exam_per_phy.min()} to {num_exam_per_phy.max()}")

# Prepare dictionary to store results
data = []
total_samples = len(df_lab)

# Count occurrences for each label
for label in label_cols:    
    counts = df_lab[label].value_counts().to_dict()
    row = {i: f"{counts.get(i, 0)} ({(counts.get(i, 0) / total_samples) * 100:.1f}%)" for i in range(len(counts))}
    data.append({**{'Image Finding': label}, **row})

# Convert dictionary to DataFrame
df_counts = pd.DataFrame(data)

# Save to CSV
df_counts.to_csv(path_out/'label_counts.csv', index=False)

print("CSV file has been saved successfully.")





ds = CXR_Dataset(path_root=path_root_pub, fold=0, split=None)
df = ds.df 
total = len(df)
total_patients = df['PatientID'].nunique()

for split in ['train', 'val', 'test']:
    df_split = df[df['Split']==split]
    num_patients = df_split['PatientID'].nunique()
    print(f"----------- {split} -----------")
    print(f"Examinations {len(df_split)} ({len(df_split)/total*100:.1f}%)")
    print(f"Patients {df_split['PatientID'].nunique()} ({df_split['PatientID'].nunique()/total_patients*100:.1f}%)")
    # print(f"Mean Age {df_split['Age'].mean()/365:.0f} ± {df_split['Age'].std()/365:.0f}")
    q1, q2 = df_split['Age'].quantile([0.025, 0.975])
    print(f"Median Age {df_split['Age'].median()/365:.0f} [{q1/365:.0f} - {q2/365:.0f}]")
    counts = df_split.groupby('PatientID')['Sex'].apply(lambda x:x.iloc[0]).value_counts()

    print("Male", counts['M'], f"({counts['M']/num_patients*100:.1f}%)")
    print("Female", counts['F'], f"({counts['F']/num_patients*100:.1f}%)")



# Set global font sizes
plt.rcParams.update({
    'font.size': 16,         # Base font size
    'axes.titlesize': 12,    # Title font size
    'axes.labelsize': 16,    # Axis label font size
    'xtick.labelsize': 16,   # X tick label size
})
# Set up the figure with subplots
num_labels = len(label_cols)
rows = (num_labels + 3) // 4  # Arrange in 4 columns
fig, axes = plt.subplots(nrows=rows, ncols=4, figsize=(16, 4 * rows))
axes = axes.flatten()

# Generate colors from the colormap - ensuring consistent ordering

# Generate pie charts
for i, label in enumerate(label_cols):
    label_values = get_labels(label)

    
    cmap_dict = {
        'HeartSize': plt.cm.Reds, 
        'PulmonaryCongestion': plt.cm.Blues, 
        'PleuralEffusion': plt.cm.Greens, 
        'PulmonaryOpacities': plt.cm.Oranges, 
        'Atelectasis': plt.cm.Purples,
    }
    cmap = cmap_dict[label.split('_')[0]]
    num_categories = 5
    colors = [cmap(0.2 + 0.7 * idx/max(1, num_categories-1)) for idx in range(num_categories)]

    
    counts = df_lab[label].value_counts().to_dict()
    values = [counts.get(idx, 0) for idx in range(len(label_values))]
    labels = [f"{label_values[idx]}" for idx in range(len(label_values))]
    
    axes[i].pie(
        values, 
        labels=labels, 
        counterclock=False,  
        explode=(0.0, 0, 0, 0) if label=="HeartSize" else (0.0, 0,0,0,0.2), 
        pctdistance=0.65, 
        autopct=lambda p: f'{p:.1f}%', 
        startangle=90, 
        colors=colors,
    )
    # axes[i].set_title(label, fontdict={'fontweight': 'bold'})
    axes[i].set_xlabel(label, fontdict={'fontweight': 'bold'})

# Hide unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout and show plot
plt.tight_layout()
plt.savefig(path_out/'label_dist_pie_chart.png',  dpi=300)



# ------------ Age ------------
# Convert age from days to years
df_lab['Age_Years'] = df_lab['Age'] / 365

# Create a 2x2 grid for the plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.style.use('seaborn-v0_8-whitegrid')

# ------------ (A) Age Distribution Histogram ------------
# Age distribution plot
axes[0, 0].hist(df_lab['Age_Years'], bins=np.linspace(0, df_lab['Age_Years'].max() * 1.05, 21), 
                  color='#E0E0E0', edgecolor='#707070', alpha=0.9) # weights=np.ones(len(df_lab)) * 100 / len(df_lab),

axes[0, 0].set_xlabel('Patient Age [Years]', fontsize=14, fontweight="bold")
axes[0, 0].set_ylabel('Radiographs [N]', fontsize=14, fontweight="bold")
axes[0, 0].set_xlim(left=0.0)
axes[0, 0].tick_params(axis='both', which='major', labelsize=12)
axes[0, 0].grid(True, linestyle='--', alpha=0.7) 
axes[0, 0].yaxis.set_major_formatter(FuncFormatter(format_with_dots))

axes[0, 0].spines["top"].set_visible(False)
axes[0, 0].spines["right"].set_visible(False)
axes[0, 0].spines["left"].set_visible(False)
axes[0, 0].spines["bottom"].set_visible(False)

axes[0, 0].set_title("(a)", fontsize=16, fontweight="bold", loc='left', x=-0.17)

# ------------ (B) Exams per Physician Bar Plot ------------
# Examinations per physician plot
num_exam_per_phy = num_exam_per_phy.sort_values(ascending=False)
axes[0, 1].bar(range(len(num_exam_per_phy)), num_exam_per_phy, color='#E0E0E0', edgecolor='#707070', alpha=0.9)
axes[0, 1].set_xlabel('Radiologist ID', fontsize=14, fontweight="bold")
axes[0, 1].set_ylabel('Radiographs [N]', fontsize=14, fontweight="bold")
axes[0, 1].tick_params(axis='both', which='major', labelsize=12)
axes[0, 1].grid(True, linestyle='--', alpha=0.7)
axes[0, 1].yaxis.set_major_formatter(FuncFormatter(format_with_dots))

axes[0, 1].spines["top"].set_visible(False)
axes[0, 1].spines["right"].set_visible(False)
axes[0, 1].spines["left"].set_visible(False)
axes[0, 1].spines["bottom"].set_visible(False)

axes[0, 1].set_title("(b)", fontsize=16, fontweight="bold", loc='left', x=-0.17)

# ------------ (C) Exams per Month Line Plot ------------
# Examinations per month plot
df_lab['YearMonth'] = df_lab['StudyDate'].dt.to_period('M')
exam_per_month = df_lab.groupby('YearMonth').size()

start_period = pd.Timestamp("2010-01-01").to_period('M')
end_period = pd.Timestamp("2023-12-31").to_period('M')
exam_per_month = exam_per_month[(exam_per_month.index >= start_period) & (exam_per_month.index <= end_period)]
exam_per_month.index = exam_per_month.index.astype(str)

axes[1, 0].plot(exam_per_month.index, exam_per_month.values, marker='o', linestyle='-', color='#E0E0E0', markerfacecolor='#707070', alpha=0.9)

# Set the x-ticks to display approximately 12 ticks
axes[1, 0].set_xticks(range(0, len(exam_per_month), max(1, len(exam_per_month)//12)))
axes[1, 0].set_xticklabels(exam_per_month.index[::max(1, len(exam_per_month)//12)], rotation=45, fontsize=12)

axes[1, 0].set_xlabel('Month', fontsize=14, fontweight="bold")
axes[1, 0].set_ylabel('Radiographs [N]', fontsize=14, fontweight="bold")
axes[1, 0].tick_params(axis='both', which='major', labelsize=12)
axes[1, 0].grid(True, linestyle='--', alpha=0.7)
axes[1, 0].yaxis.set_major_formatter(FuncFormatter(format_with_dots))

axes[1, 0].spines["top"].set_visible(False)
axes[1, 0].spines["right"].set_visible(False)
axes[1, 0].spines["left"].set_visible(False)
axes[1, 0].spines["bottom"].set_visible(False)

axes[1, 0].set_title("(c)", fontsize=16, fontweight="bold", loc='left', x=-0.17)

# ------------ (D) Gender Distribution Pie Chart ------------
labels = ['Male', 'Female']
counts = df_lab['Sex'].value_counts()
sizes = [counts.get('M', 0), counts.get('F', 0)]  

# Create the bar plot
colors = ['#E0E0E0', '#707070']
axes[1, 1].bar(labels, sizes, color=colors, edgecolor='black', alpha=0.9,  width=0.75)

# Set the x-ticks to display approximately 12 ticks
axes[1, 1].set_ylabel('Radiographs [N]', fontsize=14, fontweight="bold")
axes[1, 1].set_xlabel('Patient Sex', fontsize=14, fontweight="bold")
axes[1, 1].tick_params(axis='both', which='major', labelsize=12)
axes[1, 1].grid(True, linestyle='--', alpha=0.7)
axes[1, 1].yaxis.set_major_formatter(FuncFormatter(format_with_dots))

axes[1, 1].spines["top"].set_visible(False)
axes[1, 1].spines["right"].set_visible(False)
axes[1, 1].spines["left"].set_visible(False)
axes[1, 1].spines["bottom"].set_visible(False)

axes[1, 1].set_title("(d)", fontsize=16, fontweight="bold", loc='left', x=-0.17)

# Adjust layout to avoid overlap
plt.tight_layout()

# Save the combined plot
plt.savefig(path_out/"dataset_stats.png", dpi=300)