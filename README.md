# TAIX-Ray Code

The official codebase for the TAIX-Ray paper.

Please see our paper for a detailed description:  [TAIX-Ray Paper](https://doi.org/10.1038/s41597-026-07271-7)

<br>



## Setup
### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/TAIX-Ray.git
cd TAIX-Ray
```

### 2. Install Dependencies
Create a conda environment and install the required dependencies:
```bash
conda env create -f environment.yaml
conda activate taix-ray
```

<br>

## Dataset Preparation
### 1. Download the Dataset
The dataset can be downloaded from Hugging Face:
[TAIX-Ray Dataset](https://huggingface.co/datasets/TLAIM/TAIX-Ray)

### 2. Set Dataset Path
Update the dataset path in the following file:
```bash
cxr/data/datasets/cxr_dataset.py
```

### 3. Verify the Dataset
Run the dataset verification script:
```bash
python tests/data/test_dataset.py
```

<br>

## Training (Optional)
Train models using the following commands:

### 1. Train the Binary Classification Model
```bash
python scripts/main_train.py --task binary --model MST
```

### 2. Train the Ordinal Classification Model
```bash
python scripts/main_train.py --task ordinal --model MST --regression
```

<br>

## Evaluation
### 1. Download Pretrained Model Weights
Pretrained model checkpoints can be downloaded from:
[TAIX-Ray Models](https://huggingface.co/TLAIM/TAIX-Ray)

### 2. Evaluate the Binary Classification Model
```bash
python scripts/main_predict_binary.py --path_run path/to/checkpoint.ckpt
```

### 3. Evaluate the Ordinal Classification Model
```bash
python scripts/main_predict_ordinal.py --path_run path/to/checkpoint.ckpt
```

<br>

## Citation
If you use this work in your research, please cite:
```bibtex
@article{truhn_comprehensive_2026,
	title = {A comprehensive bedside chest radiography dataset with structured, itemized and graded radiologic reports},
	volume = {13},
	issn = {2052-4463},
	url = {https://www.nature.com/articles/s41597-026-07271-7},
	doi = {10.1038/s41597-026-07271-7},
	journal = {Scientific Data},
	author = {Truhn, Daniel and Geiger, Daniel and Siepmann, Robert and Von Der Stück, Marc Sebastian and Bressem, Keno Kyrill and Kather, Jakob Nikolas and Kuhl, Christiane and Müller-Franzes, Gustav and Nebelung, Sven},
	year = {2026},
	pages = {632},
}
```

