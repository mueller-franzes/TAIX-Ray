# TAIX-Ray Dataset

TAIX-Ray is a comprehensive dataset of about 200k bedside chest radiographs from about 50k intensive care patients at the University Hospital in Aachen, Germany, collected between 2010 and 2024. 
Trained radiologists provided structured reports at the time of acquisition, assessing key findings such as cardiomegaly, pulmonary congestion, pleural effusion, pulmonary opacities, and atelectasis on an ordinal scale. 
Please see our paper for a detailed description:  [Not yet available.](https://arxiv.org/abs/your-paper-link)

<br>

## How to Use

### Prerequisites
Ensure you have the following dependencies installed:

```bash
pip install datasets matplotlib huggingface_hub pandas tqdm
```

### Configurations
This dataset is available in two configurations. 

| **Name** | **Size** | **Image Size** |
|------------|----------|----------------|
| default    | 62GB     | 512px          |
| original   | 1.2TB    | variable       |


### Option A: Use within the Hugging Face Framework
If you want to use the dataset directly within the Hugging Face `datasets` library, you can load and visualize it as follows:

```python
from datasets import load_dataset
from matplotlib import pyplot as plt

# Load the TAIX-Ray dataset
dataset = load_dataset("TLAIM/TAIX-Ray", name="default")

# Access the training split (Fold 0)
ds_train = dataset['train']

# Retrieve a single sample from the training set
item = ds_train[0]

# Extract and display the image
image = item['Image']
plt.imshow(image, cmap='gray')
plt.savefig('image.png')  # Save the image to a file
plt.show()  # Display the image

# Print metadata (excluding the image itself)
for key in item.keys():
    if key != 'Image':
        print(f"{key}: {item[key]}")
```

### Option B: Downloading the Dataset 

If you prefer to download the dataset to a specific folder, use the following script. This will create the following folder structure:
```
.
├── data/
│   ├── 549a816ae020fb7da68a31d7d62d73c418a069c77294fc084dd9f7bd717becb9.png
│   ├── d8546c6108aad271211da996eb7e9eeabaf44d39cf0226a4301c3cbe12d84151.png
│   └── ...
└── metadata/
    ├── annoation.csv
    └── split.csv 
```

```python
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Define output paths
output_root = Path("./TAIX-Ray")

# Create folders 
data_dir = output_root / "data"
metadata_dir = output_root / "metadata"
data_dir.mkdir(parents=True, exist_ok=True)
metadata_dir.mkdir(parents=True, exist_ok=True)

# Load dataset in streaming mode
dataset = dataset = load_dataset("TLAIM/TAIX-Ray", name="default",  streaming=True)

# Process dataset
metadata = []
for split, split_dataset in dataset.items():
    print("-------- Start Download: ", split, " --------")
    for item in tqdm(split_dataset, desc="Downloading"):  # Stream data one-by-one
        uid = item["UID"]
        img = item.pop("Image")  # PIL Image object

        # Save image
        img.save(data_dir / f"{uid}.png", format="PNG")

        # Store metadata
        metadata.append(item)  

# Convert metadata to DataFrame
metadata_df = pd.DataFrame(metadata)

# Save annotations to CSV files
metadata_df.drop(columns=["Split", "Fold"]).to_csv(metadata_dir / "annotation.csv", index=False)

# Save split to CSV files (5-fold)
split_csv_path = hf_hub_download(repo_id=DATASET_NAME, repo_type="dataset", filename="split.csv", local_dir=metadata_dir)

print("Dataset streamed and saved successfully!")
```


