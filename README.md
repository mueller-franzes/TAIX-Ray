# TAIX-Ray Code

The official codebase for the TAIX-Ray paper.

Please see our paper for a detailed description:  [TAIX-Ray Paper](https://arxiv.org/abs/your-paper-link)

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
@article{yourcitation2025,
  title={TAIX-Ray: A Dataset for X-ray Classification},
  author={Your Name and Others},
  journal={Journal Name},
  year={2025}
}
```

